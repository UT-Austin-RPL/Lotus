import os
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from lotus.lifelong.algos.base import Sequential
from lotus.lifelong.metric import *
from lotus.lifelong.models import *
from lotus.lifelong.utils import *


class Multitask(Sequential):
    """
    The multitask learning baseline/upperbound.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def learn_all_tasks(self, datasets, benchmark, result_summary, use_wandb):
        self.start_task(-1)
        concat_dataset = ConcatDataset(datasets)

        # learn on all tasks, only used in multitask learning
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        train_dataloader = DataLoader(
            concat_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(concat_dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []
        all_eval_successes = []

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()
            if epoch > 0:  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )

            if use_wandb:
                wandb.log({
                    f"Training/all_task_training_loss": training_loss,
                    f"Training/all_task_training_time": (t1-t0)/60,
                    "Training/step": epoch,
                })

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                t0 = time.time()
                self.policy.eval()

                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"multitask_model_ep{epoch}.pth"
                )
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                losses.append(training_loss)

                # for multitask learning, we provide an option whether to evaluate
                # the agent once every eval_every epochs on all tasks, note that
                # this can be quite computationally expensive. Nevertheless, we
                # save the checkpoints, so users can always evaluate afterwards.
                if self.cfg.lifelong.eval_in_train:
                    success_rates = evaluate_multitask_training_success(
                        self.cfg, self, benchmark, all_tasks
                    )
                    success_rate = np.mean(success_rates)
                else:
                    success_rates = np.zeros(len(all_tasks))
                    success_rate = 0.0
                successes.append(success_rate)
                all_eval_successes.append(success_rates)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]

                if self.cfg.lifelong.eval_in_train:
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                        + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                        flush=True,
                    )
                    # plot the success rate curve to visualize the success rate on each task
                    plt.figure(figsize=(10, 5))
                    bars = plt.bar(np.arange(len(success_rates)), success_rates, align='center', alpha=0.75)
                    plt.title(f"Success Rates at Epoch {epoch}, total {success_rate}")
                    plt.xlabel("Task Index")
                    plt.ylabel("Success Rate")
                    plt.ylim(0, 1.1) 
                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')
                    

                    if use_wandb:
                        wandb.log({
                            f"Training/all_task_success_rate": success_rate,
                            f"Training/all_task_best_success_rate": prev_success_rate,
                            f"Training/all_task_AoC": tmp_successes.sum()/cumulated_counter,
                            f"Training/all_task_eval_time": (t1-t0)/60,
                            "Training/step": epoch,
                            "Training/task_suceess_rate": wandb.Image(plt),
                        })   
                    plt.close()                    

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best policy if there is any
        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        self.end_task(concat_dataset, -1, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)
        all_eval_successes = np.array(all_eval_successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
                "all_eval_successes": all_eval_successes,
            },
            auc_checkpoint_name,
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best_succ = losses[idx_at_best_succ]
            success_at_best_succ = successes[idx_at_best_succ]
            losses[idx_at_best_succ:] = loss_at_best_succ
            successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
