import argparse
import sys
import os

# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from lotus.libero import get_libero_path
from lotus.libero.benchmark import get_benchmark
from lotus.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from lotus.libero.utils.time_utils import Timer
from lotus.libero.utils.video_utils import VideoWriter, VideoWriter2
from lotus.lifelong.algos import *
from lotus.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from lotus.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from lotus.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from lotus.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        # choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["multitask_skill"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = "experiments_saved"
    return args


def main():
    args = parse_args()
    run_folder = args.experiment_dir
    all_files = os.listdir(run_folder)
    skill_model_files = [f for f in all_files if 'skill' in f and f.endswith('_model.pth')]
    try:
        if args.algo == "multitask_skill":
            meta_model_path = os.path.join(run_folder, f"meta_controller_model_ep30.pth")
            sd, cfg, previous_mask = torch_load_model(
                meta_model_path, map_location=args.device_id
            )
        else:
            # TODO: fix this
            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    skill_policies = {}
    for i in range(len(skill_model_files)):
        skill_model_path = os.path.join(run_folder, f"skill{i}_model.pth")
        skill_sd, skill_cfg, _ = torch_load_model(skill_model_path, map_location=args.device_id)
        sub_skill_policy = safe_device(SubSkill(10, skill_cfg), cfg.device)
        sub_skill_policy.policy.load_state_dict(skill_sd)
        sub_skill_policy.policy.eval()
        skill_policies[i] = sub_skill_policy.policy
    
    algo = safe_device(MetaController(10, cfg, skill_policies), cfg.device)
    algo.policy.load_state_dict(sd)


    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    task_num = benchmark.n_tasks
    descriptions = [benchmark.get_task(i).language for i in range(task_num)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)
    print("evaluating task", task.name)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[args.task_id : args.task_id + 1]
        )
    except:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_on{args.task_id}.stats",
        )
    else:
        # TODO: fix this
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
    )

    with Timer() as t, VideoWriter2(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(args.task_id)

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)
                skill_id = algo.policy.current_subtask_id
                obs, reward, done, info = env.step(actions)
                # video_writer.append_vector_obs(
                #     obs, dones, camera_name="agentview_image"
                # )
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image", skill_id=skill_id
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)


if __name__ == "__main__":
    main()
