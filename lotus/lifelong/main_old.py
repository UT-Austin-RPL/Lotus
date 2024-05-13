import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
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
from lotus.lifelong.algos import get_algo_class, get_algo_list
from lotus.lifelong.models import get_policy_list
from lotus.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
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
                new_task_name=new_task_name,
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
        manip_datasets.append(task_i_dataset)
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    gsz = cfg.data.task_group_size
    if gsz == 1:  # each manipulation task is its own lifelong learning task
        datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
        ]
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # group gsz manipulation tasks into a lifelong task, currently not used
        assert (
            n_manip_tasks % gsz == 0
        ), f"[error] task_group_size does not divide n_tasks"
        datasets = []
        n_demos = []
        n_sequences = []
        for i in range(0, n_manip_tasks, gsz):
            dataset = GroupedTaskDataset(
                manip_datasets[i : i + gsz], task_embs[i : i + gsz]
            )
            datasets.append(dataset)
            n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
            n_sequences.extend(
                [x.total_num_sequences for x in dataset.sequence_datasets]
            )

    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
            print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # prepare experiment and update the config
    create_experiment_dir(cfg, extra=cfg.exp+"_")
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

    # define lifelong algorithm
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
            print(f"[info] load pretrained model from {cfg.pretrain_model_path}")
        except:
            print(
                f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
            )
            sys.exit(0)

    # print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    # GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    # print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    if cfg.lifelong.algo == "Multitask":
        algo.train()
        s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary, cfg.use_wandb)
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
            # L = evaluate_loss(cfg, algo, benchmark, datasets)
            S = evaluate_success(
                cfg=cfg,
                algo=algo,
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

                # fig2, ax2 = plt.subplots()
                # cax2 = ax2.matshow(result_summary["L_conf_mat"], cmap=plt.cm.Reds)
                # fig2.colorbar(cax2)
                # ax2.set_title('Loss Confusion Matrix')

                # for j in range(result_summary["L_conf_mat"].shape[0]):
                #     for k in range(result_summary["L_conf_mat"].shape[1]):
                #         c = result_summary["L_conf_mat"][j,k]
                #         ax2.text(k, j, str(c), va='center', ha='center')

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
    else:
        for i in range(n_tasks):
            print(f"[info] start training on task {i}")
            algo.train()

            t0 = time.time()
            s_fwd, l_fwd = algo.learn_one_task(
                datasets[i], i, benchmark, result_summary, cfg.use_wandb
            )
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


            # evalute on all seen tasks at the end of learning each task
            if cfg.eval.eval:
                L = evaluate_loss(cfg, algo, benchmark, datasets[: i + 1])
                t2 = time.time()
                S = evaluate_success(
                    cfg=cfg,
                    algo=algo,
                    benchmark=benchmark,
                    task_ids=list(range((i + 1) * gsz)),
                    result_summary=result_summary if cfg.eval.save_sim_states else None,
                )
                t3 = time.time()
                result_summary["L_conf_mat"][i][: i + 1] = L
                result_summary["S_conf_mat"][i][: i + 1] = S

                if cfg.use_wandb:
                    fig1, ax1 = plt.subplots()
                    cax1 = ax1.matshow(result_summary["S_conf_mat"], cmap=plt.cm.Blues)
                    fig1.colorbar(cax1)
                    ax1.set_title('Success Confusion Matrix')

                    for j in range(result_summary["S_conf_mat"].shape[0]):
                        for k in range(result_summary["S_conf_mat"].shape[1]):
                            c = result_summary["S_conf_mat"][j,k]
                            ax1.text(k, j, str(c), va='center', ha='center')

                    fig2, ax2 = plt.subplots()
                    cax2 = ax2.matshow(result_summary["L_conf_mat"], cmap=plt.cm.Reds)
                    fig2.colorbar(cax2)
                    ax2.set_title('Loss Confusion Matrix')

                    for j in range(result_summary["L_conf_mat"].shape[0]):
                        for k in range(result_summary["L_conf_mat"].shape[1]):
                            c = result_summary["L_conf_mat"][j,k]
                            ax2.text(k, j, str(c), va='center', ha='center')

                    wandb.log({
                        "Summary/success_confusion_matrix": wandb.Image(fig1),
                        "Summary/loss_confusion_matrix": wandb.Image(fig2),
                        "Summary/task_step": i,
                        "Summary/eval_loss_time": (t2-t1)/60,
                        "Summary/eval_success_time": (t3-t2)/60,
                        f"Summary/task_{i}_losses": wandb.Histogram(L, num_bins=len(L)),
                        f"Summary/task_{i}_success_rates": wandb.Histogram(S, num_bins=len(S)),
                    })

                    plt.close(fig1)
                    plt.close(fig2)

                    wandb.run.summary["success_confusion_matrix"] = result_summary[
                        "S_conf_mat"
                    ]
                    wandb.run.summary["loss_confusion_matrix"] = result_summary[
                        "L_conf_mat"
                    ]
                    wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                    wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
                    # wandb.run.summary.update() # this is not needed in training


                print(
                    f"[info] train time (min) {(t1-t0)/60:.1f} "
                    + f"eval loss time {(t2-t1)/60:.1f} "
                    + f"eval success time {(t3-t2)/60:.1f}"
                )
                print(("[Task %2d loss ] " + " %4.2f |" * (i + 1)) % (i, *L))
                print(("[Task %2d succ.] " + " %4.2f |" * (i + 1)) % (i, *S))
                torch.save(
                    result_summary, os.path.join(cfg.experiment_dir, f"result.pt")
                )

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
