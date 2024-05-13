import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py
import numpy as np
import cv2
import os

import argparse
import init_path
from models.model_utils import safe_cuda
from models.conf_utils import *
from models.torch_utils import *
from policy_learning.models import *
from policy_learning.datasets import *
from policy_learning.path_templates import *

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
import json
from hydra.experimental import compose, initialize
import pprint
from torch.utils.tensorboard import SummaryWriter
import kornia

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    
    folder_path = cfg.folder
    
    modalities = cfg.repr.modalities    
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)
    use_gmm = True
    print("Using GMM: ", use_gmm)

    if cfg.skill_training.random_affine:
        data_aug = torch.nn.Sequential(*[torch.nn.ReplicationPad2d(cfg.skill_training.affine_translate),
                                               kornia.augmentation.RandomCrop((128, 128))])
        # affine_translate = 0.03 if "affine_translate" not in cfg.skill_training else cfg.skill_training.affine_translate
        # transform = transforms.RandomAffine(degrees=0, translate=(affine_translate, affine_translate))
    # else:
    transform = None

    subgoal_embedding_file_name = None
    if cfg.skill_training.policy_type == "no_subgoal":
        policy_type = PolicyType.NO_SUBGOAL
    elif cfg.skill_training.policy_type == "normal_subgoal":
        policy_type = PolicyType.NORMAL_SUBGOAL
    # elif cfg.skill_training.policy_type == "vae_subgoal":
    #     policy_type = PolicyType.VAE_SUBGOAL
    #     subgoal_embedding_file_name = folder_path + f"results/{cfg.dataset_name}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.policy_type}_embedding.hdf5"        

    meta_dataset = BCMetaDataset(data_file_name=folder_path+f"datasets/{cfg.data.dataset_name}/demo.hdf5",
                                 subtasks_file_name=folder_path+f"skill_data/{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5",
                                 subgoal_embedding_file_name=subgoal_embedding_file_name,
                                 use_rnn=cfg.skill_training.use_rnn,
                                 subtask_id=cfg.skill_training.subtask_id,
                                 data_modality=cfg.skill_training.data_modality,
                                 use_eye_in_hand=cfg.skill_training.use_eye_in_hand,
                                 policy_type=policy_type,
                                 subgoal_cfg=cfg.skill_subgoal_cfg,
                                 skill_training_cfg=cfg.skill_training,
                                 gripper_smoothing=cfg.skill_training.gripper_smoothing,
                                 transform=transform,
                                 rnn_horizon=cfg.skill_subgoal_cfg.horizon)

    output_parent_dir = output_parent_dir_template(cfg)
    os.makedirs(output_parent_dir, exist_ok=True)
    training_cfg = EasyDict()
    training_cfg.skill_training = cfg.skill_training
    training_cfg.skill_subgoal_cfg = cfg.skill_subgoal_cfg
    with open(f"{output_parent_dir}/skill_cfg.json", "w") as f:
        json.dump(training_cfg, f, cls=NpEncoder, indent=4)
    
    
    for i in range(meta_dataset.num_subtasks):
        dataset = meta_dataset.get_dataset(idx=i)
        if dataset is None:
            continue
        print(f"Subtask id: {dataset.subtask_id}")
        dataloader = DataLoader(dataset, batch_size=cfg.skill_training.batch_size, shuffle=True, num_workers=cfg.skill_training.num_workers)

        action_dim = dataset.action_dim
        env_name = dataset.env_name

        # template = subskill_path_template(cfg, subtask_id=i, use_cvae=cfg.skill_cvae_cfg.enable)
        template = subskill_path_template(cfg, subtask_id=i, use_changepoint=cfg.skill_training.use_changepoint)
        output_dir = template.output_dir
        model_checkpoint_name = template.model_checkpoint_name
        summary_writer_name = template.summary_writer_name

        os.makedirs(output_dir, exist_ok=True)
        writer = SummaryWriter(summary_writer_name)

        if cfg.skill_subgoal_cfg is not None:
            subgoal_visual_feature_dimension = cfg.skill_subgoal_cfg.visual_feature_dimension
            use_subgoal_eye_in_hand = cfg.skill_subgoal_cfg.use_eye_in_hand
        else:
            subgoal_visual_feature_dimension = 0
            use_subgoal_eye_in_hand = False

        if use_gmm:
            bc_policy = safe_cuda(BCGMMPolicy(action_dim=action_dim,
                                       state_dim=cfg.skill_training.state_dim,
                                       proprio_dim=dataset.proprio_dim,
                                       data_modality=cfg.skill_training.data_modality,
                                       use_eye_in_hand=cfg.skill_training.use_eye_in_hand,
                                       use_subgoal_eye_in_hand=use_subgoal_eye_in_hand,
                                       activation=cfg.skill_training.activation,
                                       z_dim=cfg.repr.z_dim,
                                       num_kp=cfg.skill_training.num_kp,
                                       img_h=cfg.skill_training.img_h,
                                       img_w=cfg.skill_training.img_w,
                                       visual_feature_dimension=cfg.skill_training.visual_feature_dimension,
                                       subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                       action_squash=cfg.skill_training.action_squash,
                                       policy_layer_dims=cfg.skill_training.policy_layer_dims,
                                       policy_type=policy_type,
                                       subgoal_type=cfg.skill_subgoal_cfg.subgoal_type))
        else:
            bc_policy = safe_cuda(BCPolicy(action_dim=action_dim,
                                        state_dim=cfg.skill_training.state_dim,
                                        proprio_dim=dataset.proprio_dim,
                                        data_modality=cfg.skill_training.data_modality,
                                        use_eye_in_hand=cfg.skill_training.use_eye_in_hand,
                                        use_subgoal_eye_in_hand=use_subgoal_eye_in_hand,
                                        activation=cfg.skill_training.activation,
                                        z_dim=cfg.repr.z_dim,
                                        num_kp=cfg.skill_training.num_kp,
                                        img_h=cfg.skill_training.img_h,
                                        img_w=cfg.skill_training.img_w,
                                        visual_feature_dimension=cfg.skill_training.visual_feature_dimension,
                                        subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                        action_squash=cfg.skill_training.action_squash,
                                        policy_layer_dims=cfg.skill_training.policy_layer_dims,
                                        policy_type=policy_type,
                                        subgoal_type=cfg.skill_subgoal_cfg.subgoal_type))

        if cfg.use_checkpoint:
            policy_state_dict, _ = torch_load_model(model_checkpoint_name)
            bc_policy.load_state_dict(policy_state_dict)
            print("loaded checkpoint")

        optimizer = torch.optim.Adam(bc_policy.parameters(), lr=cfg.skill_training.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=40)

        if cfg.skill_training.use_rnn:
            reduction = cfg.skill_training.rnn_loss_reduction
        else:
            reduction = "sum"
        
        mse_loss = torch.nn.MSELoss(reduction=reduction)
        testing_mse_loss = torch.nn.MSELoss(reduction="sum")
        bce_loss = torch.nn.BCELoss(reduction=reduction)
        prev_training_loss = None
        training_loss = None
        testing_loss = None
        if use_gmm:
            training_loss_thresh = -1e10
        else:
            training_loss_thresh = 0.01
        epoch = 0
        counter = 0

        writer_image_written = False
        writer_graph_written = False
        while testing_loss is None or (epoch < cfg.skill_training.num_epochs and testing_loss > training_loss_thresh):
        # for epoch in range(cfg.num_epochs):
            bc_policy.train()
            training_loss = 0

            action = None
            training_kl_loss = 0
            target_aciton = None
            for (idx, data) in enumerate(dataloader):
                target_action = data["action"]
                data["state_image"] = data_aug(data["state_image"])
                # Writing images to tensorboard
                if not writer_image_written:
                    img_grid = torchvision.utils.make_grid(data["state_image"][:, :3, ...])
                    writer.add_image('agent', img_grid)
                    img_grid = torchvision.utils.make_grid(data["state_image"][:, 3:, ...])
                    writer.add_image('eye in hand', img_grid)
                    img_grid = torchvision.utils.make_grid(data["subgoal"][: ,:3, ...])
                    writer.add_image('subgoal', img_grid)
                    writer_image_written = True
                
                # img = np.zeros((256, 128, 3))
                # img[:128 ,:, :] = (data["obs_seq"][0, 0, :3, ...].detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
                # img[128: ,:, :] = (data["subgoal"][0, ...].detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
                
                # cv2.imwrite(f"debugging_image/{i}_{idx}.png", img)
                # if not writer_graph_written:
                #     bc_policy.eval()
                #     writer.add_graph(bc_policy, data)
                #     writer_graph_written = True
                #     bc_policy.train()

                if use_gmm:
                    dist = bc_policy(data)
                    loss = bc_policy._policy_layers.loss_fn(dist, target_action, "mean")

                else:
                    action = bc_policy(data)
                    loss = mse_loss(target_action, action)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
            writer.add_scalar("training_loss",
                              training_loss,
                              epoch)


            if epoch % 5 == 0:
                testing_loss = 0
                losses = []
                bc_policy.eval()
                with torch.no_grad():
                    for data in dataloader:
                        target_action = data["action"]

                        action = bc_policy(data)
                        if use_gmm:
                            dist = bc_policy(data)
                            loss = bc_policy._policy_layers.loss_fn(dist, target_action, "mean")
                        else:
                            action = bc_policy(data)
                            loss = testing_mse_loss(target_action, action)
                        testing_loss += loss.item()
                        losses.append(loss.item())

                if prev_training_loss is None:
                    prev_training_loss = testing_loss
                if prev_training_loss >= testing_loss:
                    torch_save_model(bc_policy, model_checkpoint_name, cfg=cfg)
                    prev_training_loss = testing_loss
                    counter = 0
                writer.add_scalar("testing_loss",
                                  testing_loss,
                                  epoch)
            if epoch % 20 == 0:
                print(f"Training loss: {np.round(training_loss, 3)}, Kl loss: {np.round(training_kl_loss, 3)}")
                print(f"Testing loss: {testing_loss}, max: {np.max(losses)}, min: {np.min(losses)}, mean: {np.mean(losses)}")
                counter += 1
                
            if optimizer.param_groups[0]['lr'] > cfg.skill_training.min_lr:
                scheduler.step(testing_loss)
            epoch += 1
        del bc_policy
        

if __name__ == "__main__":
    main()
    
