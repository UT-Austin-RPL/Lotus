import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# libero_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# lifelong_dir = os.path.dirname(os.path.abspath(__file__))
# if libero_dir not in sys.path:
#     sys.path.insert(0, libero_dir)
# if lifelong_dir not in sys.path:
#     sys.path.insert(0, lifelong_dir)
import re
import json
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from lotus.libero import get_libero_path
from lotus.libero.benchmark import get_benchmark
# from lotus.lifelong.algos import get_algo_class, get_algo_list
from lotus.lifelong.algos import *
from lotus.lifelong.models import get_policy_list
from lotus.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset, SkillLearningDataset, MetaPolicyDataset, MetaPolicySequenceDataset
from lotus.lifelong.metric import evaluate_loss, evaluate_success
from lotus.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)

import matplotlib.pyplot as plt
import glob
import h5py
import init_path
from skill_learning.models.model_utils import safe_cuda
from skill_learning.models.conf_utils import *
from skill_learning.models.torch_utils import *

def get_subtask_label(idx, saved_ep_subtasks_seq, horizon):
    for (start_idx, end_idx, subtask_label) in saved_ep_subtasks_seq:
        if start_idx <= idx <= end_idx:
            return min(end_idx, idx + horizon), subtask_label

def save_subgoal_embedding(cfg, networks, data_file_name_list, skill_learning_cfg, skill_exp_name):
    subgoal_embedding_file_name = os.path.join(cfg.experiment_dir, f"subgoal_embedding.hdf5")
    subgoal_embedding_file = h5py.File(subgoal_embedding_file_name, "w")
    for data_file_name in data_file_name_list:
        dataset_category, dataset_name = data_file_name.split("/")[1:]
        dataset_name = dataset_name.split(".")[0]
        demo_file = h5py.File(f"{data_file_name}", "r")
        file_pattern = f"skill_learning/results/{skill_exp_name}/skill_data/{dataset_category}/{dataset_name}*"
        matching_files = glob.glob(file_pattern)
        subtasks_file_name = matching_files[0]
        subtask_file = h5py.File(subtasks_file_name, "r")
        
        if cfg.goal_modality == "dinov2":
            dinov2_feature_file_name = re.sub(r"(datasets/)([^/]+)(/)", r"\1dinov2/\2\3", data_file_name)
            dinov2_feature_file = h5py.File(dinov2_feature_file_name, "r")

        demo_num = len(demo_file['data'].keys())
        grp = subgoal_embedding_file.create_group(f"{dataset_name}")
        for ep_idx in range(demo_num):
            # Generate embedding
            if f"demo_subtasks_seq_{ep_idx}" not in subtask_file["subtasks"]:
                continue
            saved_ep_subtasks_seq = subtask_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
            agentview_images = demo_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()]
            eye_in_hand_images = demo_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()]
            ee_states = demo_file[f"data/demo_{ep_idx}/obs/ee_states"][()]
            gripper_states = demo_file[f"data/demo_{ep_idx}/obs/gripper_states"][()]
            joint_states = demo_file[f"data/demo_{ep_idx}/obs/joint_states"][()]
            if cfg.goal_modality == "dinov2":
                dinov2_embedding = dinov2_feature_file[f"data/demo_{ep_idx}/embedding"][()]


            embeddings = []
            for i in range(len(agentview_images)):
                future_idx, subtask_label = get_subtask_label(i, saved_ep_subtasks_seq, horizon=skill_learning_cfg.skill_subgoal_cfg.horizon)
                agentview_image = safe_cuda(torch.from_numpy(np.array(agentview_images[future_idx]).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.
                eye_in_hand_image = safe_cuda(torch.from_numpy(np.array(eye_in_hand_images[future_idx]).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.

                if cfg.goal_modality == "BUDS":
                    if skill_learning_cfg.skill_subgoal_cfg.use_eye_in_hand:
                        state_image = torch.cat([agentview_image, eye_in_hand_image], dim=1)
                    else:
                        state_image = agentview_image
                    embedding = networks[subtask_label].get_embedding(state_image).detach().cpu().numpy().squeeze()
                elif cfg.goal_modality == "ee_states":
                    embedding = np.concatenate([ee_states[future_idx], gripper_states[future_idx]])
                elif cfg.goal_modality == "joint_states":
                    embedding = np.concatenate([joint_states[future_idx], gripper_states[future_idx]])
                elif cfg.goal_modality == "dinov2":
                    embedding = dinov2_embedding[future_idx]
                # import ipdb; ipdb.set_trace()
                embeddings.append(embedding)

            ep_data_grp = grp.create_group(f"demo_{ep_idx}")
            ep_data_grp.create_dataset("embedding", data=np.stack(embeddings, axis=0))
        subtask_file.close()
        demo_file.close()
        grp.attrs["embedding_dim"] = len(embeddings[-1])
    subgoal_embedding_file.close()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks
    new_task_name = benchmark.new_task_name

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        # manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions) # (n_tasks, emb_dim)
    benchmark.set_task_embs(task_embs)
    task_names = benchmark.get_task_names()

    modalities = cfg.skill_learning.repr.modalities    
    modality_str = get_modalities_str(cfg.skill_learning)
    skill_learning_cfg = cfg.skill_learning
    skill_exp_name = skill_learning_cfg.exp_name
    exp_dir = f"skill_learning/results/{skill_exp_name}/skill_data"
    data_file_name_list = []
    subtasks_file_name_list = []
    used_data_file_name_list_skill = task_names #[]
    used_data_file_name_list_meta = task_names
    for dataset_category in os.listdir(exp_dir):
        dataset_category_path = os.path.join(exp_dir, dataset_category)
        if os.path.isdir(dataset_category_path) and dataset_category in ['libero_object','libero_spactial','libero_goal', "libero_10", "libero_90", "rw_all"]:
            for dataset_name in os.listdir(dataset_category_path):
                dataset_name = dataset_name.split("_demo_")[0] + '_demo'
                data_file_name_list.append(f"datasets/{dataset_category}/{dataset_name}.hdf5")
                file_pattern = f"skill_learning/results/{skill_exp_name}/skill_data/{dataset_category}/{dataset_name}*"
                matching_files = glob.glob(file_pattern)
                assert len(matching_files)==1
                subtasks_file_name_list.append(matching_files[0])

    skill_dataset = SkillLearningDataset(data_file_name_list=data_file_name_list,
                                 subtasks_file_name_list=subtasks_file_name_list,
                                 subtask_id=[],
                                 data_modality=skill_learning_cfg.skill_training.data_modality,
                                 use_eye_in_hand=skill_learning_cfg.skill_training.use_eye_in_hand,
                                 subgoal_cfg=skill_learning_cfg.skill_subgoal_cfg,
                                 seq_len=cfg.data.seq_len,
                                 task_embs=task_embs,
                                 goal_modality=cfg.goal_modality,
                                 new_task_name=new_task_name,
                                 demo_range=range(0, 50),
                                 used_data_file_name_list=used_data_file_name_list_skill)

    gsz = cfg.data.task_group_size

    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
            print(f"        {benchmark.get_task(i*gsz+j).language}")
    # print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    # print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg, skill_exp_name=skill_exp_name, extra=cfg.exp+"_")
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg)
        wandb.run.name = cfg.experiment_name

    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
        "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
        "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    }

    if cfg.eval.save_sim_states:
        # for saving the evaluate simulation states, so we can replay them later
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # for testing task p when the agent learns to task k
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # for testing task k at the e-th epoch when the agent learns on task k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]

    if cfg.lifelong.algo == "Multitask_Skill":
        # skill policy training
        skill_policies = {}
        for i in range(skill_dataset.num_subtasks):
            dataset = skill_dataset.get_dataset(idx=i)
            print(f"Subtask id: {i}")
            sub_skill_policy = safe_device(SubSkill(n_tasks, cfg), cfg.device)
            if cfg.pretrain_model_path != "":
                sub_skill_policy.load_skill(skill_id=i, experiment_dir=cfg.pretrain_model_path)
            # sub_skill_policy.eval()
            if dataset is None:
                print(f"No Data on Subtask {i}")
            else:
                if i in skill_dataset.train_dataset_id:
                    sub_skill_policy.train()
                    loss = sub_skill_policy.learn_one_skill(dataset, benchmark, result_summary, i, cfg.use_wandb)
            skill_policies[i] = sub_skill_policy.policy
            del sub_skill_policy

        # set eval mode for all skill policies
        for skill_policy in skill_policies.values():
            skill_policy.eval()
        
        del skill_policy
        del skill_dataset
        del dataset

        # save the subgoal embedding
        save_subgoal_embedding(cfg, skill_policies, data_file_name_list, skill_learning_cfg, skill_exp_name)

        # meta policy training

        meta_dataset = MetaPolicySequenceDataset(data_file_name_list=data_file_name_list,
                                        embedding_file_name=os.path.join(cfg.experiment_dir, f"subgoal_embedding.hdf5"),
                                        subtasks_file_name_list=subtasks_file_name_list,
                                        use_eye_in_hand=skill_learning_cfg.meta.use_eye_in_hand,
                                        task_names = task_names, # include task order infos
                                        task_embs=task_embs,
                                        new_task_name=new_task_name,
                                        demo_range=range(0, 50),
                                        used_data_file_name_list=used_data_file_name_list_meta)

        cfg.skill_learning.num_subtasks = meta_dataset.num_subtasks
        cfg.skill_learning.subgoal_embedding_dim = meta_dataset.subgoal_embedding_dim
        # save the experiment config file, so we can resume or replay later
        with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
            json.dump(cfg, f, cls=NpEncoder, indent=4)
        meta_policy = safe_device(MetaController(n_tasks, cfg, skill_policies), cfg.device)
        meta_policy.train()

        if cfg.pretrain_model_path != "":
            meta_policy.load_meta_policy(experiment_dir=cfg.pretrain_model_path)

        s_fwd, l_fwd, kl_loss, ce_loss, embedding_loss = meta_policy.learn_multi_task(meta_dataset, benchmark, result_summary, cfg.use_wandb)
        result_summary["L_fwd"][-1] = l_fwd
        result_summary["S_fwd"][-1] = s_fwd

        if cfg.use_wandb:
            fig1, ax1 = plt.subplots()
            bars1 = ax1.bar(np.arange(len(result_summary["S_fwd"])), result_summary["S_fwd"])
            ax1.set_title('Forward Transfer Success')

            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

            fig2, ax2 = plt.subplots()
            bars2 = ax2.bar(np.arange(len(result_summary["L_fwd"])), result_summary["L_fwd"])
            ax2.set_title('Forward Transfer Loss')

            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

            wandb.log({
                "Summary/fwd_transfer_success": wandb.Image(fig1),
                "Summary/fwd_transfer_loss": wandb.Image(fig2),
            })

            plt.close(fig1)
            plt.close(fig2)


        # evalute on all seen tasks at the end if eval.eval is true
        if cfg.eval.eval:
            # L = evaluate_loss(cfg, meta_policy, benchmark, datasets)
            S = evaluate_success(
                cfg=cfg,
                algo=meta_policy,
                benchmark=benchmark,
                task_ids=list(range(n_manip_tasks)),
                result_summary=result_summary if cfg.eval.save_sim_states else None,
            )

            # result_summary["L_conf_mat"][-1] = L
            result_summary["S_conf_mat"][-1] = S

            if cfg.use_wandb:
                fig1, ax1 = plt.subplots()
                cax1 = ax1.matshow(result_summary["S_conf_mat"], cmap=plt.cm.Blues)
                fig1.colorbar(cax1)
                ax1.set_title('Success Confusion Matrix')

                for j in range(result_summary["S_conf_mat"].shape[0]):
                    for k in range(result_summary["S_conf_mat"].shape[1]):
                        c = result_summary["S_conf_mat"][j,k]
                        ax1.text(k, j, str(c), va='center', ha='center')

                wandb.log({
                    "Summary/success_confusion_matrix": wandb.Image(fig1),
                    # "Summary/loss_confusion_matrix": wandb.Image(fig2),
                    # f"Summary/all_task_losses": wandb.Histogram(L, num_bins=len(L)),
                    f"Summary/all_task_success_rates": wandb.Histogram(S, num_bins=len(S)),
                })

                plt.close(fig1)
                # plt.close(fig2)

                wandb.run.summary["success_confusion_matrix"] = result_summary[
                    "S_conf_mat"
                ]
                # wandb.run.summary["loss_confusion_matrix"] = result_summary[
                #     "L_conf_mat"
                # ]
                wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                # wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
                # wandb.run.summary.update() # this is not needed in training


            # print(("[All task loss ] " + " %4.2f |" * n_tasks) % tuple(L))
            print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(S))

            torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
    
    elif cfg.lifelong.algo == "Singletask_Skill":
        for i in range(n_tasks):
            print(f"[info] start training on task {i}")
            t0 = time.time()
            benchmark_name = benchmark.name
            task_description = benchmark.get_task(i).language
            print(f"Task description: {benchmark_name} - {task_description}")

            # skill policy training
            skill_policies = {}
            for j in range(skill_dataset.num_subtasks):
                dataset = skill_dataset.get_dataset(idx=j)
                if dataset is None:
                    continue
                print(f"Subtask id: {dataset.subtask_id}")
                sub_skill_policy = safe_device(SubSkill(n_tasks, cfg), cfg.device)
                sub_skill_policy.train()
                loss = sub_skill_policy.learn_one_skill(dataset, benchmark, result_summary, j, cfg.use_wandb)
                skill_policies[j] = sub_skill_policy.policy

            del skill_dataset
            del dataset

            # save the subgoal embedding
            save_subgoal_embedding(cfg, skill_policies, data_file_name_list, skill_learning_cfg, skill_exp_name)
            # del skill_policies
            del sub_skill_policy

            # meta policy training

            meta_dataset = MetaPolicyDataset(data_file_name_list=data_file_name_list,
                                            embedding_file_name=os.path.join(cfg.experiment_dir, f"subgoal_embedding.hdf5"),
                                            subtasks_file_name_list=subtasks_file_name_list,
                                            use_eye_in_hand=skill_learning_cfg.meta.use_eye_in_hand)

            cfg.skill_learning.num_subtasks = meta_dataset.num_subtasks
            cfg.skill_learning.subgoal_embedding_dim = meta_dataset.subgoal_embedding_dim
            meta_policy = safe_device(MetaController(n_tasks, cfg, skill_policies), cfg.device)
            meta_policy.train()

            s_fwd, l_fwd, kl_loss, ce_loss, embedding_loss = meta_policy.learn_one_task(meta_dataset, i, benchmark, result_summary, cfg.use_wandb)
            result_summary["S_fwd"][i] = s_fwd
            result_summary["L_fwd"][i] = l_fwd
            t1 = time.time()

            if cfg.use_wandb:
                fig1, ax1 = plt.subplots()
                bars1 = ax1.bar(np.arange(len(result_summary["S_fwd"])), result_summary["S_fwd"])
                ax1.set_title('Forward Transfer Success')

                for bar in bars1:
                    yval = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

                fig2, ax2 = plt.subplots()
                bars2 = ax2.bar(np.arange(len(result_summary["L_fwd"])), result_summary["L_fwd"])
                ax2.set_title('Forward Transfer Loss')

                for bar in bars2:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')
                wandb.log({
                    "Summary/fwd_transfer_success": wandb.Image(fig1),
                    "Summary/fwd_transfer_loss": wandb.Image(fig2),
                    "Summary/task_step": i,
                    "Summary/train_time": (t1-t0)/60,
                })

                plt.close(fig1)
                plt.close(fig2)
            break
    else:
        raise NotImplementedError

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
