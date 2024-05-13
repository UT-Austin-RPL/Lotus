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

def get_subtask_label(idx, saved_ep_subtasks_seq, horizon):
    for (start_idx, end_idx, subtask_label) in saved_ep_subtasks_seq:
        if start_idx <= idx <= end_idx:
            return min(end_idx, idx + horizon), subtask_label

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize

@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    
    folder_path = "./"
    
    modalities = cfg.repr.modalities    
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)
    use_gmm = True
    print("Using GMM: ", use_gmm)

    out_parent_dir = f"datasets/{cfg.data.dataset_name}"

    demo_file_name = os.path.join(out_parent_dir, "demo.hdf5")
    demo_file = h5py.File(demo_file_name, "r")
    num_eps = demo_file["data"].attrs["num_eps"]

    action_dim = len(demo_file["data/ep_0/actions"][()][0])

    proprio_dim = 0
    if cfg.skill_training.use_gripper:
        proprio_dim += len(demo_file["data/ep_0/gripper_states"][()][0]) * 5
    if cfg.skill_training.use_joints:
        proprio_dim += len(demo_file["data/ep_0/joint_states"][()][0])

    if not cfg.skill_training.use_changepoint:
        subtask_file_name = folder_path+f"skill_data/{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"
    else:
        subtask_file_name = folder_path+f"results/skill_data/cp_{cfg.data.dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}.hdf5"
        
    subtask_file = h5py.File(subtask_file_name, "r")
    num_subtask = subtask_file["subtasks"].attrs["num_subtasks"]

    if not cfg.skill_training.use_changepoint:
        subgoal_embedding_file_name = subgoal_embedding_path_template(cfg, modality_str)
    else:
        subgoal_embedding_file_name = cp_subgoal_embedding_path_template(cfg, modality_str)        
        
    subgoal_embedding_file = h5py.File(subgoal_embedding_file_name, "w")

    networks = {}
    if cfg.skill_subgoal_cfg is not None:
        subgoal_visual_feature_dimension = cfg.skill_subgoal_cfg.visual_feature_dimension
    else:
        raise ValueError

    if cfg.skill_training.policy_type == "no_subgoal":
        policy_type = PolicyType.NO_SUBGOAL
    elif cfg.skill_training.policy_type == "normal_subgoal":
        policy_type = PolicyType.NORMAL_SUBGOAL
    elif cfg.skill_training.policy_type == "vae_subgoal":
        policy_type = PolicyType.VAE_SUBGOAL
    if cfg.skill_training.policy_type == "normal_subgoal":
        for i in range(num_subtask):
            # template = subskill_path_template(cfg, subtask_id=i, use_cvae=cfg.skill_cvae_cfg.enable, use_changepoint=cfg.skill_training.use_changepoint)
            template = subskill_path_template(cfg, subtask_id=i, use_changepoint=cfg.skill_training.use_changepoint)
            output_dir = template.output_dir
            model_checkpoint_name = template.model_checkpoint_name
            network_state_dict, network_cfg = torch_load_model(model_checkpoint_name)
            print(network_cfg)

            if use_gmm:
                network = safe_cuda(BCGMMPolicy(action_dim=action_dim,
                                            state_dim=network_cfg.skill_training.state_dim,
                                            proprio_dim=proprio_dim,
                                            data_modality=network_cfg.skill_training.data_modality,
                                            use_eye_in_hand=network_cfg.skill_training.use_eye_in_hand,
                                            use_subgoal_eye_in_hand=network_cfg.skill_subgoal_cfg.use_eye_in_hand,
                                            activation=network_cfg.skill_training.activation,
                                            z_dim=network_cfg.repr.z_dim,
                                            num_kp=network_cfg.skill_training.num_kp,
                                            img_h=network_cfg.skill_training.img_h,
                                            img_w=network_cfg.skill_training.img_w,
                                            visual_feature_dimension=network_cfg.skill_training.visual_feature_dimension,
                                            subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                            action_squash=network_cfg.skill_training.action_squash,
                                            policy_layer_dims=network_cfg.skill_training.policy_layer_dims,
                                            policy_type=policy_type,
                                            subgoal_type=cfg.skill_subgoal_cfg.subgoal_type))
            else:
                network = safe_cuda(BCPolicy(action_dim=action_dim,
                                            state_dim=network_cfg.skill_training.state_dim,
                                            proprio_dim=proprio_dim,
                                            data_modality=network_cfg.skill_training.data_modality,
                                            use_eye_in_hand=network_cfg.skill_training.use_eye_in_hand,
                                            use_subgoal_eye_in_hand=network_cfg.skill_subgoal_cfg.use_eye_in_hand,
                                            activation=network_cfg.skill_training.activation,
                                            z_dim=network_cfg.repr.z_dim,
                                            num_kp=network_cfg.skill_training.num_kp,
                                            img_h=network_cfg.skill_training.img_h,
                                            img_w=network_cfg.skill_training.img_w,
                                            visual_feature_dimension=network_cfg.skill_training.visual_feature_dimension,
                                            subgoal_visual_feature_dimension=subgoal_visual_feature_dimension,
                                            action_squash=network_cfg.skill_training.action_squash,
                                            policy_layer_dims=network_cfg.skill_training.policy_layer_dims,
                                            policy_type=policy_type,
                                            subgoal_type=cfg.skill_subgoal_cfg.subgoal_type))
            network.load_state_dict(network_state_dict)
            networks[i] = network

    grp = subgoal_embedding_file.create_group("data")
    for ep_idx in range(num_eps):
        # Generate embedding
        if f"ep_subtasks_seq_{ep_idx}" not in subtask_file["subtasks"]:
            continue
        saved_ep_subtasks_seq = subtask_file["subtasks"][f"ep_subtasks_seq_{ep_idx}"][()]
        agentview_image_names = demo_file[f"data/ep_{ep_idx}/agentview_image_names"][()]
        eye_in_hand_image_names = demo_file[f"data/ep_{ep_idx}/eye_in_hand_image_names"][()]

        embeddings = []
        print("Ep: ", ep_idx)
        for i in range(len(agentview_image_names)):
            future_idx, subtask_label = get_subtask_label(i, saved_ep_subtasks_seq, horizon=cfg.skill_subgoal_cfg.horizon)

            # if cfg.subgoal_cfg.use_final_goal:
            if cfg.skill_subgoal_cfg.use_final_goal:
                future_idx = len(agent_view_image_names) - 1
            
            if cfg.skill_training.policy_type == "vae_subgoal":
                # For vae, there is no subtask id for vae embedding.
                subtask_label = 0
            agentview_image = safe_cuda(torch.from_numpy(np.array(Image.open(agentview_image_names[future_idx])).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.
            eye_in_hand_image = safe_cuda(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[future_idx])).transpose(2, 0, 1)).unsqueeze(0)).float() / 255.

            if network_cfg.skill_subgoal_cfg.use_eye_in_hand:
                state_image = torch.cat([agentview_image, eye_in_hand_image], dim=1)
            else:
                state_image = agentview_image
            embedding = networks[subtask_label].get_embedding(state_image).detach().cpu().numpy().squeeze()
            embeddings.append(embedding)

        if ep_idx % 10 == 0:
            for (start_idx, end_idx, subtask_label) in saved_ep_subtasks_seq:
                print(f"Subtask: {subtask_label}")
                print(np.round(embeddings[start_idx], 2))
                print(np.round(embeddings[end_idx], 2))
            
        ep_data_grp = grp.create_group(f"ep_{ep_idx}")
        ep_data_grp.create_dataset("embedding", data=np.stack(embeddings, axis=0))

    grp.attrs["embedding_dim"] = len(embeddings[-1])
    subtask_file.close()
    demo_file.close()
    subgoal_embedding_file.close()
    

if __name__ == "__main__":
    main()
