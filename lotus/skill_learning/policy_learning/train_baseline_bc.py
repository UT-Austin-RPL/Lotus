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

    dataset = BaselineBCDataset(data_file_name=folder_path + f"datasets/{cfg.data.dataset_name}/demo.hdf5",
                                data_modality=cfg.skill_training.data_modality,
                                use_eye_in_hand=True,
                                use_subgoal_eye_in_hand=False,                 
                                subgoal_cfg=None,
                                transform=None,
                                skill_training_cfg=cfg.skill_training,                 
                                baseline_type="single_skill")
    # import ipdb; ipdb.set_trace()
    meta_policy = safe_cuda(BaselineBCPolicy(action_dim=dataset.action_dim,
                                        state_dim=cfg.skill_training.state_dim,
                                        proprio_dim=dataset.proprio_dim,
                                        data_modality=cfg.skill_training.data_modality,
                                        use_eye_in_hand=True,
                                        use_subgoal_eye_in_hand=False,
                                        use_subgoal_spatial_softmax=True,
                                        use_goal=False,
                                        activation='relu',
                                        action_squash=True,
                                        z_dim=128,
                                        num_kp=64,
                                        img_h=128,
                                        img_w=128,
                                        visual_feature_dimension=64,
                                        subgoal_visual_feature_dimension=0,
                                        policy_layer_dims=[256, 256]))

    if not cfg.skill_training.use_changepoint:
        template = meta_path_template(cfg)
    else:
        template = cp_meta_path_template(cfg)
    output_dir = template.output_dir
    #model_name = template.model_name
    #summary_writer_name = template.summary_writer_name
    model_name = f"{output_dir}/BC.pth"
    summary_writer_name = f"{output_dir}/BC"
    os.makedirs(f"{output_dir}", exist_ok=True)
    print(f"Model initialized!, {model_name}")
            
    if cfg.use_checkpoint:
        meta_state_dict, _ = torch_load_model(model_name)
        meta_policy.load_state_dict(meta_state_dict)
        print("loaded checkpoint")

    dataloader = DataLoader(dataset, batch_size=cfg.meta.batch_size, shuffle=True, num_workers=cfg.meta.num_workers)

    env_name = dataset.env_name

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
        for data in dataloader:

            if cfg.meta.random_affine:
                data["state_image"] = data_aug(data["state_image"])

            action = meta_policy(data)

            action_loss = mse_loss(data["action"], action)

            loss = action_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        writer.add_scalar("loss", training_loss, epoch)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} / {cfg.meta.num_epochs}, Training loss: {training_loss}")


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
    
