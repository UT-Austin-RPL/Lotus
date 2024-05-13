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



class SubSkill(Sequential):
    """
    The SubSkill policy for skill learning
    learning algorithms.
    """
    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)
        if cfg.goal_modality == "BUDS":
            cfg.shape_meta["all_shapes"]["subgoal"] = [3, 128, 128]
        elif cfg.goal_modality == "ee_states":
            cfg.shape_meta["all_shapes"]["subgoal"] = [8]
        elif cfg.goal_modality == "joint_states":
            cfg.shape_meta["all_shapes"]["subgoal"] = [9]
        elif cfg.goal_modality == "dinov2":
            cfg.shape_meta["all_shapes"]["subgoal"] = [768] # [1536]
        self.policy = BCTransformerSkillPolicy(cfg, cfg.shape_meta) #TODO: update

    def start_task(self, task):
        super().start_task(task)
    
    def learn_one_skill(self, dataset, benchmark, result_summary, skill_id, use_wandb):
        self.start_task(-1)

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"skill{skill_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=0, #self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )
        # start training
        print(f"[info] start training skill {skill_id}")
        prev_training_loss = None
        losses = []
        cumulated_counter = 0.0
        for epoch in range(0, self.cfg.train.n_epochs + 1):
        # for epoch in range(0, 15 + 1):
            t0 = time.time()
            if epoch > 0 or (self.cfg.pretrain):  # update
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
                    f"Skill_Training/skill{skill_id}_training_loss": training_loss,
                    f"Skill_Training/skill{skill_id}_training_time": (t1-t0)/60,
                    "Skill_Training/step": epoch,
                })

            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                t0 = time.time()
                self.policy.eval()
                losses.append(training_loss)

                testing_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    testing_loss += loss
                testing_loss /= len(train_dataloader)

                if prev_training_loss is None:
                    prev_training_loss = testing_loss
                if prev_training_loss >= testing_loss:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_training_loss = testing_loss

                t1 = time.time()
                cumulated_counter += 1.0
            
            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # load the best policy if there is any
        # if self.cfg.lifelong.eval_in_train:
        #     self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])


        # return the metrics regarding skill_training
        losses = np.array(losses)

        return losses.sum() / cumulated_counter
    
    def load_skill(self, skill_id, experiment_dir):
        model_checkpoint_name = os.path.join(
            experiment_dir, f"skill{skill_id}_model.pth"
        )
        if os.path.exists(model_checkpoint_name):
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        model_checkpoint_save_name = os.path.join(
            self.experiment_dir, f"skill{skill_id}_model.pth"
        )
        torch_save_model(self.policy, model_checkpoint_save_name, cfg=self.cfg)
        # self.policy.eval()



class MetaController(Sequential):
    """
    The MetaController policy for skill learning
    learning algorithms.
    """
    def __init__(self, n_tasks, cfg, skill_policies):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)
        cfg.shape_meta["all_shapes"] = {}
        cfg.shape_meta["all_shapes"]["agentview_rgb"] = [3, 128, 128]
        if cfg.goal_modality == "BUDS":
            subgoal_embedding_dim = cfg.skill_learning.subgoal_embedding_dim
        elif cfg.goal_modality == "ee_states":
            subgoal_embedding_dim = 8
        elif cfg.goal_modality == "joint_states":
            subgoal_embedding_dim = 9
        elif cfg.goal_modality == "dinov2":
            subgoal_embedding_dim = 768 #1536 #TODO: update
        self.policy = MetaCVAETransformerPolicy(cfg=cfg,
                                    num_subtasks=cfg.skill_learning.num_subtasks,
                                    subgoal_embedding_dim=subgoal_embedding_dim,
                                    id_layer_dims=cfg.skill_learning.meta.id_layer_dims,
                                    embedding_layer_dims=cfg.skill_learning.meta.embedding_layer_dims,
                                    use_eye_in_hand=cfg.skill_learning.meta.use_eye_in_hand,
                                    activation=cfg.skill_learning.meta.activation,
                                    # use_skill_id_in_encoder=cfg.skill_learning.meta_cvae_cfg.use_skill_id,
                                    subgoal_type=cfg.skill_learning.skill_subgoal_cfg.subgoal_type,
                                    latent_dim=cfg.skill_learning.meta_cvae_cfg.latent_dim,
                                    policy_type=cfg.skill_learning.skill_training.policy_type,
                                    use_spatial_softmax=cfg.skill_learning.meta.use_spatial_softmax,
                                    num_kp=cfg.skill_learning.meta.num_kp,
                                    visual_feature_dimension=cfg.skill_learning.meta.visual_feature_dimension,
                                    kl_coeff=cfg.skill_learning.meta_cvae_cfg.kl_coeff,
                                    skill_policies = skill_policies,)
        self.skill_policies = skill_policies

    def start_task(self, task):
        super().start_task(task)
    
    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss, ce_loss, embedding_loss, kl_loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item(), ce_loss.item(), embedding_loss.item(), kl_loss.item()
    
    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss, ce_loss, embedding_loss, kl_loss = self.policy.compute_loss(data)
        return loss.item(), ce_loss.item(), embedding_loss.item(), kl_loss.item()

    def learn_multi_task(self, dataset, benchmark, result_summary, use_wandb):
        self.start_task(-1)

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"meta_controller_model.pth"
        )
        all_tasks = list(range(benchmark.n_tasks))

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=0, #self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # start training
        print(f"[info] start training meta controller")
        prev_training_loss = None
        losses = []
        kl_losses = []
        ce_losses = []
        embedding_losses = []
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        all_eval_successes = []
        for epoch in range(0, self.cfg.train.n_epochs + 1):
        # for epoch in range(0, 0 + 1):

            t0 = time.time()
            if epoch > 0 or (self.cfg.pretrain):  # update
                self.policy.train()
                training_loss = 0.0
                training_kl_loss = 0.0
                training_embedding_loss = 0.0
                training_ce_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss, ce_loss, embedding_loss, kl_loss = self.observe(data)
                    training_loss += loss
                    training_ce_loss += ce_loss
                    training_embedding_loss += embedding_loss
                    training_kl_loss += kl_loss
                # training_loss /= len(train_dataloader)
                # training_ce_loss /= len(train_dataloader)
                # training_embedding_loss /= len(train_dataloader)
                # training_kl_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                training_ce_loss = 0.0
                training_embedding_loss = 0.0
                training_kl_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss, ce_loss, embedding_loss, kl_loss = self.eval_observe(data)
                    training_loss += loss
                    training_ce_loss += ce_loss
                    training_embedding_loss += embedding_loss
                    training_kl_loss += kl_loss
                # training_loss /= len(train_dataloader)
                # training_ce_loss /= len(train_dataloader)
                # training_embedding_loss /= len(train_dataloader)
                # training_kl_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | Train loss: {training_loss:5.2f} | "
                f"\nTraining ce loss: {training_ce_loss:5.2f} | Training embedding loss: {training_embedding_loss:5.2f} | "
                f"Training kl loss: {training_kl_loss:5.2f} | Time: {(t1-t0)/60:4.2f}"
            )


            if use_wandb:
                wandb.log({
                    f"MetaPolicy_Training/all_task_training_loss": training_loss,
                    f"MetaPolicy_Training/all_task_training_ce_loss": training_ce_loss,
                    f"MetaPolicy_Training/all_task_training_embedding_loss": training_embedding_loss,
                    f"MetaPolicy_Training/all_task_training_kl_loss": training_kl_loss,
                    f"MetaPolicy_Training/all_task_training_time": (t1-t0)/60,
                    "MetaPolicy_Training/step": epoch,
                })
            
            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                t0 = time.time()
                self.policy.eval()
                model_checkpoint_name_ep = os.path.join(
                    self.experiment_dir, f"meta_controller_model_ep{epoch}.pth"
                )
                torch_save_model(self.policy, model_checkpoint_name_ep, cfg=self.cfg)
                
                losses.append(training_loss)
                kl_losses.append(training_kl_loss)
                ce_losses.append(training_ce_loss)
                embedding_losses.append(training_embedding_loss)

                # #evaluate the loss on training dataset
                # testing_loss = 0.0
                # for (idx, data) in enumerate(train_dataloader):
                #     loss, ce_loss, embedding_loss, kl_loss = self.eval_observe(data)
                #     testing_loss += loss
                # testing_loss /= len(train_dataloader)

                # if prev_training_loss is None:
                #     prev_training_loss = testing_loss
                # if prev_training_loss >= testing_loss:
                #     torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                #     prev_training_loss = testing_loss

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

                if prev_success_rate < success_rate and (not self.cfg.pretrain):
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
                            f"MetaPolicy_Training/all_task_success_rate": success_rate,
                            f"MetaPolicy_Training/all_task_best_success_rate": prev_success_rate,
                            f"MetaPolicy_Training/all_task_AoC": tmp_successes.sum()/cumulated_counter,
                            f"MetaPolicy_Training/all_task_eval_time": (t1-t0)/60,
                            "MetaPolicy_Training/step": epoch,
                            "MetaPolicy_Training/task_suceess_rate": wandb.Image(plt),
                        })
                    plt.close()
            
            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()
        
        # load the best policy if there is any
        if self.cfg.lifelong.eval_in_train:
            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])


        # return the metrics regarding skill_training
        losses = np.array(losses)
        kl_losses = np.array(kl_losses)
        ce_losses = np.array(ce_losses)
        embedding_losses = np.array(embedding_losses)
        successes = np.array(successes)
        all_eval_successes = np.array(all_eval_successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"multitask_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "all_eval_successes": all_eval_successes,
                "loss": losses,
                "kl_loss": kl_losses,
                "ce_loss": ce_losses,
                "embedding_loss": embedding_losses,
            },
            auc_checkpoint_name,
        )

        if self.cfg.lifelong.eval_in_train:
            loss_at_best_succ = losses[idx_at_best_succ]
            success_at_best_succ = successes[idx_at_best_succ]
            losses[idx_at_best_succ:] = loss_at_best_succ
            successes[idx_at_best_succ:] = success_at_best_succ
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter, kl_losses.sum() / cumulated_counter, ce_losses.sum() / cumulated_counter, embedding_losses.sum() / cumulated_counter


    def learn_one_task(self, dataset, task_id, benchmark, result_summary, use_wandb):
        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_meta_controller_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=0, #self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            # persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # currently save the best model

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        prev_training_loss = None
        losses = []
        kl_losses = []
        ce_losses = []
        embedding_losses = []
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):
        # for epoch in range(0, 0 + 1):

            t0 = time.time()
            if epoch > 0 or (self.cfg.pretrain):  # update
                self.policy.train()
                training_loss = 0.0
                training_kl_loss = 0.0
                training_embedding_loss = 0.0
                training_ce_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss, ce_loss, embedding_loss, kl_loss = self.observe(data)
                    training_loss += loss
                    training_ce_loss += ce_loss
                    training_embedding_loss += embedding_loss
                    training_kl_loss += kl_loss
                # training_loss /= len(train_dataloader)
                # training_ce_loss /= len(train_dataloader)
                # training_embedding_loss /= len(train_dataloader)
                # training_kl_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                training_ce_loss = 0.0
                training_embedding_loss = 0.0
                training_kl_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss, ce_loss, embedding_loss, kl_loss = self.eval_observe(data)
                    training_loss += loss
                    training_ce_loss += ce_loss
                    training_embedding_loss += embedding_loss
                    training_kl_loss += kl_loss
                # training_loss /= len(train_dataloader)
                # training_ce_loss /= len(train_dataloader)
                # training_embedding_loss /= len(train_dataloader)
                training_kl_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | Train loss: {training_loss:5.2f} | "
                f"\nTraining ce loss: {training_ce_loss:5.2f} | Training embedding loss: {training_embedding_loss:5.2f} | "
                f"Training kl loss: {training_kl_loss:5.2f} | Time: {(t1-t0)/60:4.2f}"
            )

            if use_wandb:
                wandb.log({
                    f"MetaPolicy_Training/task_{task_id}_training_loss": training_loss,
                    f"MetaPolicy_Training/task_{task_id}_training_ce_loss": training_ce_loss,
                    f"MetaPolicy_Training/task_{task_id}_training_embedding_loss": training_embedding_loss,
                    f"MetaPolicy_Training/task_{task_id}_training_kl_loss": training_kl_loss,
                    f"MetaPolicy_Training/task_{task_id}_training_time": (t1-t0)/60,
                    "MetaPolicy_Training/step": epoch,
                })
            
            if epoch % self.cfg.eval.eval_every == 0:  # evaluate BC loss
                t0 = time.time()
                self.policy.eval()
                losses.append(training_loss)
                kl_losses.append(training_kl_loss)
                ce_losses.append(training_ce_loss)
                embedding_losses.append(training_embedding_loss)

                # #evaluate the loss on training dataset
                # testing_loss = 0.0
                # for (idx, data) in enumerate(train_dataloader):
                #     loss, ce_loss, embedding_loss, kl_loss = self.eval_observe(data)
                #     testing_loss += loss
                # testing_loss /= len(train_dataloader)

                # if prev_training_loss is None:
                #     prev_training_loss = testing_loss
                # if prev_training_loss >= testing_loss:
                #     torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                #     prev_training_loss = testing_loss
                
                # single task evaluation
                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str="",
                )
        
                successes.append(success_rate)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        f"MetaPolicy_Training/task_{task_id}_success_rate": success_rate,
                        f"MetaPolicy_Training/task_{task_id}_best_success_rate": prev_success_rate,
                        f"MetaPolicy_Training/task_{task_id}_AoC": tmp_successes.sum()/cumulated_counter,
                        f"MetaPolicy_Training/task_{task_id}_eval_time": (t1-t0)/60,
                        "MetaPolicy_Training/step": epoch,
                    })
            
            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()
        
        # load the best performance agent on the current task
        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        # end learning the current task, some algorithms need post-processing
        # self.end_task(dataset, task_id, benchmark)


        # return the metrics regarding skill_training
        losses = np.array(losses)
        kl_losses = np.array(kl_losses)
        ce_losses = np.array(ce_losses)
        embedding_losses = np.array(embedding_losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
                "kl_loss": kl_losses,
                "ce_loss": ce_losses,
                "embedding_loss": embedding_losses,
            },
            auc_checkpoint_name,
        )

        # pretend that the agent stops learning once it reaches the peak performance
        losses[idx_at_best_succ:] = losses[idx_at_best_succ]
        successes[idx_at_best_succ:] = successes[idx_at_best_succ]

        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter, kl_losses.sum() / cumulated_counter, ce_losses.sum() / cumulated_counter, embedding_losses.sum() / cumulated_counter

    def load_meta_policy(self, experiment_dir):
        model_checkpoint_name = os.path.join(
            experiment_dir, "meta_controller_model_ep50.pth"
        )
        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])
        # self.policy.eval()