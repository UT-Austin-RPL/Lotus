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
from models.torch_utils import *
from models.conf_utils import *
from policy_learning.models import *
from policy_learning.datasets import *
from policy_learning.path_templates import *

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize
from torch.utils.tensorboard import SummaryWriter
import kornia

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(hydra_cfg):

    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    
    modalities = cfg.repr.modalities
    modality_str= get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)
    suffix_str = ""

    folder_path = "./"
    if cfg.meta.random_affine:
        data_aug = torch.nn.Sequential(*[torch.nn.ReplicationPad2d(cfg.meta.affine_translate),
                                               kornia.augmentation.RandomCrop((128, 128))])
    transform = None
    cfg.meta.use_embedding = False
    if not cfg.meta.use_rnn:
        if not cfg.skill_training.policy_type == "no_subgoal":
            if not cfg.skill_training.use_changepoint:
                embedding_file_name = subgoal_embedding_path_template(cfg, modality_str)
            else:
                embedding_file_name = cp_subgoal_embedding_path_template(cfg, modality_str)
        else:
            embedding_file_name = None

        if not cfg.skill_training.use_changepoint:
            subtasks_file_name = f"skill_data/{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"
        else:
            subtasks_file_name = f"results/skill_data/cp_{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"            
        dataset = MetaPolicyDataset(data_file_name=folder_path + f"datasets/{cfg.data.dataset_name}/demo.hdf5",
                                    embedding_file_name=embedding_file_name,
                                    subtasks_file_name=subtasks_file_name,
                                    use_embedding=cfg.meta.use_embedding,
                                    use_eye_in_hand=cfg.meta.use_eye_in_hand,
                                    transform=transform)

        cfg.subgoal_embedding_dim = dataset.subgoal_embedding_dim
        cfg.num_subtasks = dataset.num_subtasks

        if cfg.meta_cvae_cfg.enable:
            if not cfg.meta.separate_id_prediction:
                meta_policy = safe_cuda(MetaCVAEPolicy(num_subtasks=cfg.num_subtasks,
                                                       subgoal_embedding_dim=cfg.subgoal_embedding_dim,
                                                       id_layer_dims=cfg.meta.id_layer_dims,
                                                       embedding_layer_dims=cfg.meta.embedding_layer_dims,
                                                       use_eye_in_hand=cfg.meta.use_eye_in_hand,
                                                       activation=cfg.meta.activation,
                                                   # use_skill_id_in_encoder=cfg.meta_cvae_cfg.use_skill_id,
                                                   subgoal_type=cfg.skill_subgoal_cfg.subgoal_type,
                                                   latent_dim=cfg.meta_cvae_cfg.latent_dim,
                                                   policy_type=cfg.skill_training.policy_type,
                                                   use_spatial_softmax=cfg.meta.use_spatial_softmax,
                                                   num_kp=cfg.meta.num_kp,
                                                   visual_feature_dimension=cfg.meta.visual_feature_dimension))

            else:
                meta_policy = safe_cuda(MetaSeparateCVAEPolicy(num_subtasks=cfg.num_subtasks,
                                                       subgoal_embedding_dim=cfg.subgoal_embedding_dim,
                                                       id_layer_dims=cfg.meta.id_layer_dims,
                                                       embedding_layer_dims=cfg.meta.embedding_layer_dims,
                                                       use_eye_in_hand=cfg.meta.use_eye_in_hand,
                                                       activation=cfg.meta.activation,
                                                       # use_skill_id_in_encoder=cfg.meta_cvae_cfg.use_skill_id,
                                                       subgoal_type=cfg.skill_subgoal_cfg.subgoal_type,
                                                       latent_dim=cfg.meta_cvae_cfg.latent_dim,
                                                       policy_type=cfg.skill_training.policy_type,
                                                       use_spatial_softmax=cfg.meta.use_spatial_softmax,
                                                       num_kp=cfg.meta.num_kp,
                                                       visual_feature_dimension=cfg.meta.visual_feature_dimension))
        else:
            meta_policy = safe_cuda(MetaPolicy(num_subtasks=cfg.num_subtasks,
                                                   subgoal_embedding_dim=cfg.subgoal_embedding_dim,
                                                   id_layer_dims=cfg.meta.id_layer_dims,
                                                   embedding_layer_dims=cfg.meta.embedding_layer_dims,
                                                   use_eye_in_hand=cfg.meta.use_eye_in_hand,
                                                   activation=cfg.meta.activation,
                                                   subgoal_type=cfg.skill_subgoal_cfg.subgoal_type,
                                                   latent_dim=cfg.meta_cvae_cfg.latent_dim,
                                                   policy_type=cfg.skill_training.policy_type,
                                                   use_spatial_softmax=cfg.meta.use_spatial_softmax,
                                                   num_kp=cfg.meta.num_kp,
                                                   visual_feature_dimension=cfg.meta.visual_feature_dimension))

    if not cfg.skill_training.use_changepoint:
        template = meta_path_template(cfg)
    else:
        template = cp_meta_path_template(cfg)
    output_dir = template.output_dir
    model_name = template.model_name
    summary_writer_name = template.summary_writer_name
    os.makedirs(f"{output_dir}", exist_ok=True)
    print(f"Model initialized!, {model_name}")
            
    if cfg.use_checkpoint:
        meta_state_dict, _ = torch_load_model(model_name)
        meta_policy.load_state_dict(meta_state_dict)
        print("loaded checkpoint")

    dataloader = DataLoader(dataset, batch_size=cfg.meta.batch_size, shuffle=True, num_workers=cfg.meta.num_workers)

    num_subtasks = dataset.num_subtasks
    env_name = dataset.env_name
    # if cfg.skill_training.policy_type == "no_subgoal":
    #     output_dir = f"results/{cfg.data.dataset_name}/meta_policy_{modality_str}_{cfg.data.z_dim}_{cfg.skill_training.policy_type}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.agglomoration.K}_{cfg.agglomoration.affinity}"
    # else:
    #     output_dir = f"results/{cfg.dataset_name}/meta_policy_{modality_str}_{cfg.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"

    optimizer = torch.optim.Adam(meta_policy.parameters(), lr=cfg.meta.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=50)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    mse_loss = torch.nn.MSELoss(reduction='sum')
    prev_training_loss = None

    writer = SummaryWriter(summary_writer_name)
    
    output_parent_dir = output_parent_dir_template(cfg)
    training_cfg = EasyDict()
    training_cfg.meta = cfg.meta
    training_cfg.meta_cvae_cfg = cfg.meta_cvae_cfg
    # with open(f"{output_parent_dir}/meta_cfg.json", "w") as f:
    #     json.dump(training_cfg, f, cls=NpEncoder, indent=4)

    writer_graph_written = False
    for epoch in range(cfg.meta.num_epochs):
        meta_policy.train()
        training_loss = 0
        training_kl_loss = 0
        total_embedding_loss = 0
        target_aciton = None
        for (inp_data, target_data) in dataloader:

            if cfg.meta.random_affine:
                inp_data["state_image"] = data_aug(inp_data["state_image"])
            if cfg.meta_cvae_cfg.enable:
                subtask_id_prediction, subgoal_embedding, mu, logvar = meta_policy(inp_data)

                ce_loss = cross_entropy_loss(subtask_id_prediction.view(-1, num_subtasks), target_data["id"].view(-1))
                embedding_loss = mse_loss(target_data["embedding"], subgoal_embedding)

                kl_loss = -cfg.meta_cvae_cfg.kl_coeff * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)
                loss = ce_loss + embedding_loss + kl_loss

            else:
                subtask_id_prediction, subgoal_embedding = meta_policy(inp_data)

                ce_loss = cross_entropy_loss(subtask_id_prediction.view(-1, num_subtasks), target_data["id"].view(-1))
                embedding_loss = mse_loss(target_data["embedding"], subgoal_embedding)
                
                kl_loss = 0
                loss = ce_loss + embedding_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if not writer_graph_written:
            #     writer_graph_written = True
            #     writer.add_graph(meta_policy, inp_data)

            training_loss += loss.item()
            if cfg.meta_cvae_cfg.enable:
                training_kl_loss += kl_loss.item()
            total_embedding_loss += embedding_loss.item()

        writer.add_scalar("kl_loss", training_kl_loss, epoch)
        writer.add_scalar("embedding", total_embedding_loss, epoch)
        writer.add_scalar("loss", training_loss, epoch)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} / {cfg.meta.num_epochs}, Training loss: {training_loss}, kl loss: {training_kl_loss}, Embedding: {total_embedding_loss}")
            # print(mu, logvar)
        # if epoch % 50 == 0 and epoch > 0:
        #     import pdb; pdb.set_trace()
        # print(f"{subtask_id_prediction.view(-1, num_subtasks)}")
        # print(f"{target_subtask_id.view(-1)}" )

        if prev_training_loss is None:
            prev_training_loss = training_loss
        if prev_training_loss > training_loss or epoch % 20 == 0:
            torch_save_model(meta_policy, model_name, cfg=cfg)
            prev_training_loss = training_loss

        # if optimizer.param_groups[0]['lr'] > 1e-6:
        #     scheduler.step(training_loss)
        # else:
        #     break


if __name__ == "__main__":
    main()
    
