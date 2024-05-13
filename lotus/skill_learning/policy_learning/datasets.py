import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from models.model_utils import *
import init_path
from PIL import Image
import cv2

from policy_learning.models import PolicyType
from models.torch_utils import to_onehot

class SubtaskDataset(Dataset):
    def __init__(self,
                 data_file,
                 subtask_file,
                 subgoal_embedding_file,
                 subtask_id,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 policy_type=PolicyType.NO_SUBGOAL,
                 gripper_smoothing=False,
                 subgoal_cfg=None,
                 skill_training_cfg=None,
                 transform=None,
                 use_final_goal=False,
                 skip_task=[]):
        """

        Args:
           data_modality (list): provide a list of data modality. "image" - using image; "proprio": incorporating gripper pose + gripper position; "state": using low-dim states
           subgoal_cfg (EasyDict): if None, no subgoal is used; or the subgoal_cfg is defined 
        """
        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.use_subgoal_eye_in_hand = use_subgoal_eye_in_hand
        self.transform = transform
        
        self.env_name = data_file["data"].attrs["env"]
        self.subtask_id = subtask_id

        self.policy_type = policy_type
        self.subgoal_cfg = subgoal_cfg
        self.skill_training_cfg = skill_training_cfg

        subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
            

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.states = []
        self.actions = []
        self.proprios = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.goal_images = []
        self.subgoal_indices = []

        action_threshold = -1
        smooth_window = 10
        before_window = int(smooth_window * 0.3)
        after_window = int(smooth_window * 0.7)
        skip_action_indices = []

        skip_ep_indices = []
        if skip_task != []:
            for idx in skip_task:
                ep_indices = data_file["data/task"].attrs[f"{idx}"]
                skip_ep_indices += ep_indices.tolist()
            print("Skipping : ", skip_ep_indices, len(skip_ep_indices))

        for (i, start_idx, end_idx) in subtask_segmentation:
            if i in skip_ep_indices:
                continue
            actions = data_file[f"data/ep_{i}/actions"][()][start_idx:end_idx+1]

            
            for j in range(len(actions)):
                if gripper_smoothing:
                    action_history = list(actions[max(0, j- before_window):min(j+after_window, len(actions))][:, -1])
                    if j - smooth_window < 0:
                        action_history += [actions[0][-1]] * (abs(j-before_window))
                    elif j + smooth_window > len(actions):
                        action_history += [actions[-1][-1]] * (abs(j+after_window - len(actions)))
                    smoothed_action = np.mean(action_history)
                    self.actions.append(np.concatenate([actions[j][:-1], [smoothed_action]]))
                else:
                    self.actions.append(actions[j])
        self.actions = np.array(self.actions)
        self.total_len = len(self.actions)
        self.actions = safe_cuda(torch.from_numpy(self.actions).float())

        self.policy_type = policy_type


        if policy_type == PolicyType.NORMAL_SUBGOAL:
            self.subgoal_images = []
            count = 0
            for (i, start_idx, end_idx) in subtask_segmentation:
                if i in skip_ep_indices:
                    continue
                
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()][start_idx:end_idx+1]
                for j in range(len(agentview_image_names)):
                    future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                    self.subgoal_indices.append(future_idx + count)

                count = len(self.subgoal_indices)
            #         if np.linalg.norm(data_file[f"data/ep_{i}/actions"][()][start_idx+j][:-1]) <= action_threshold:
            #             continue
            #         self.subgoal_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[future_idx])).transpose(2, 0, 1)))

            # self.subgoal_images = safe_cuda(torch.stack(self.subgoal_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(max(self.subgoal_indices) == len(self.actions)-1)

        # elif policy_type == PolicyType.VAE_SUBGOAL:
        #     vae_embedding_file = subgoal_embedding_file
        #     self.vae_embeddings = []
        #     for (i, start_idx, end_idx) in subtask_segmentation:
        #         vae_embeddings = vae_embedding_file[f"data/ep_{i}/embedding"][()][start_idx:end_idx+1]
        #         for j in range(len(vae_embeddings)):
        #             if np.linalg.norm(data_file[f"data/ep_{i}/actions"][()][start_idx+j][:-1]) <= action_threshold:
        #                 continue
        #             self.vae_embeddings.append(torch.from_numpy(vae_embeddings[j]))
        #     self.vae_embeddings = safe_cuda(torch.stack(self.vae_embeddings, dim=0)).float()
        

        if "image" in data_modality:
            for (i, start_idx, end_idx) in subtask_segmentation:
                if i in skip_ep_indices:
                    continue
                
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()][start_idx:end_idx+1]
                if self.use_eye_in_hand:
                    eye_in_hand_image_names = data_file[f"data/ep_{i}/eye_in_hand_image_names"][()][start_idx:end_idx+1]
                for j in range(len(agentview_image_names)):
                    if np.linalg.norm(data_file[f"data/ep_{i}/actions"][()][start_idx+j][:-1]) <= action_threshold:
                        continue
                    
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    if self.use_eye_in_hand:
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))

            self.agentview_images =safe_cuda(torch.stack(self.agentview_images, dim=0))
            if self.use_eye_in_hand:
                self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))

            assert(len(self.actions) == len(self.agentview_images))

            
        if "proprio" in data_modality:
            gripper_states_list = []
            joint_states_list = []
            for (i, start_idx, end_idx) in subtask_segmentation:
                if i in skip_ep_indices:
                    continue

                gripper_states = data_file[f"data/ep_{i}/gripper_states"][()]
                if self.skill_training_cfg.use_gripper:
                    for j in range(start_idx, end_idx+1):
                        gripper_state = []
                        for k in range(j-5, j):
                            if k < 0:
                                gripper_state += gripper_states[0].tolist()
                            else:
                                gripper_state += gripper_states[k].tolist()
                        gripper_states_list.append(gripper_state)
                
                if self.skill_training_cfg.use_joints:
                    joint_states = torch.from_numpy(data_file[f"data/ep_{i}/joint_states"][()][start_idx:end_idx+1])
                    for j in range(len(joint_states)):
                        joint_states_list.append(joint_states[j])


            if self.skill_training_cfg.use_gripper and self.skill_training_cfg.use_joints:
                self.proprios = safe_cuda(torch.cat([torch.stack(joint_states_list, dim=0),
                                                     torch.tensor(gripper_states_list)], dim=1)).float()
            elif self.skill_training_cfg.use_gripper:
                self.proprios = safe_cuda(torch.tensor(gripper_states_list)).float()
            elif self.skill_training_cfg.use_joints:
                self.proprios = safe_cuda(torch.stack(joint_states_list, dim=0)).float()
            assert(len(self.proprios) == len(self.actions))
                
        # if "state" in data_modality:
        #     # low dimensional state training
        #     for (i, start_idx, end_idx) in subtask_segmentation:
        #         low_dim_states = torch.from_numpy(data_file[f"data/ep_{i}/low_dim_states"][()][start_idx:end_idx + 1])
        #         self.states.append(low_dim_states)
        #     self.states = safe_cuda(torch.cat(self.states, dim=0)).float()

        #     assert(len(self.actions) == len(self.states))

        print("Dataset info: ")
        print("Action dim: ", self.action_dim)
        print("Proprio dim: ", self.proprio_dim)
    @property
    def action_dim(self):
        return self.actions.shape[-1]

    @property
    def proprio_dim(self):
        if self.proprios == []:
            return 0
        else:
            return self.proprios.shape[-1]
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        action = self.actions[idx, ...]

        data = {"action": action}
        if "image" in self.data_modality:
            agentview_image = self.agentview_images[idx].float() / 255.
            if self.use_eye_in_hand:
                eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
                state_image = torch.cat((agentview_image, eye_in_hand_image), dim=0)
                if self.transform is not None:
                    state_image = self.transform(state_image)
                data["state_image"] = state_image
            else:
                if self.transform is not None:
                    state_image = self.transform(state_image)
                data["state_image"] = agentview_image
        if "proprio" in self.data_modality:
            data["proprio"] = self.proprios[idx]
        if "state" in self.data_modality:
            data["state"] = self.states[idx]

        if self.policy_type == PolicyType.NORMAL_SUBGOAL:
            if self.use_subgoal_eye_in_hand:
                subgoal_image = torch.cat((self.agentview_images[self.subgoal_indices[idx]].float() / 255.,
                                           self.eye_in_hand_images[self.subgoal_indices[idx]].float() / 255.))
            else:
                subgoal_image = self.agentview_images[self.subgoal_indices[idx]].float() / 255.
            # subgoal_image = self.subgoal_images[idx].float() / 255.

            data["subgoal"] = subgoal_image
        elif self.policy_type == PolicyType.VAE_SUBGOAL:
            data["vae_embedding"] = self.vae_embeddings[idx]
        return data

class BCMetaDataset():
    def __init__(self,
                 data_file_name,
                 subtasks_file_name,
                 subgoal_embedding_file_name,
                 use_rnn=False,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 policy_type=PolicyType.NO_SUBGOAL,
                 subgoal_cfg=None,
                 skill_training_cfg=None,
                 subtask_id=[],
                 gripper_smoothing=False,
                 transform=None,
                 rnn_horizon=0,
                 skip_task_id=[]):
        self.f = h5py.File(data_file_name, "r")
        self.subtasks_f = h5py.File(subtasks_file_name, "r")
        if subgoal_embedding_file_name is not None:
            self.subgoal_embedding_f = h5py.File(subgoal_embedding_file_name, "r")
        else:
            self.subgoal_embedding_f = None

        self.use_rnn = use_rnn
        self.subtask_id = subtask_id

        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.transform = transform
        self.num_subtasks = self.subtasks_f["subtasks"].attrs["num_subtasks"]

        self.policy_type = policy_type
        self.subgoal_cfg = subgoal_cfg
        self.skill_training_cfg = skill_training_cfg

        self.gripper_smoothing = gripper_smoothing

        self.rnn_horizon = rnn_horizon
        self.skip_task_id = skip_task_id

        print("Score of this data is: ", self.subtasks_f["subtasks"].attrs["score"])
        self.datasets = []

    def get_dataset(self, idx):
        if self.subtask_id != []:
            if idx not in self.subtask_id:
                return None

        if not self.use_rnn:
            dataset = SubtaskDataset(self.f,
                                     self.subtasks_f,
                                     self.subgoal_embedding_f,
                                     idx,
                                     data_modality=self.data_modality,
                                     use_eye_in_hand=self.use_eye_in_hand,
                                     use_subgoal_eye_in_hand=self.subgoal_cfg.use_eye_in_hand,
                                     policy_type=self.policy_type,
                                     subgoal_cfg=self.subgoal_cfg,
                                     skill_training_cfg=self.skill_training_cfg,
                                     gripper_smoothing=self.gripper_smoothing,
                                     transform=self.transform,
                                     skip_task=self.skip_task_id)

        else:
            print("Using RNN")
            dataset = SubtaskSequenceDataset(self.f,
                                             self.subtasks_f,
                                             self.subgoal_embedding_f,
                                             idx,
                                             data_modality=self.data_modality,
                                             use_eye_in_hand=self.use_eye_in_hand,
                                             use_subgoal_eye_in_hand=self.subgoal_cfg.use_eye_in_hand,
                                             policy_type=self.policy_type,
                                             subgoal_cfg=self.subgoal_cfg,
                                             skill_training_cfg=self.skill_training_cfg,
                                             gripper_smoothing=self.gripper_smoothing,
                                             transform=self.transform,
                                             rnn_horizon=self.rnn_horizon)

        print(idx, len(dataset))
        return dataset
    
    def close(self):
        self.f.close()
        self.subtasks_f.close()


class SubtaskSequenceDataset(Dataset):
    def __init__(self,
                 data_file,
                 subtask_file,
                 subgoal_embedding_file,
                 subtask_id,
                 data_modality=["image", "proprio"],                 
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=True,
                 policy_type=PolicyType.NO_SUBGOAL,
                 gripper_smoothing=False,
                 subgoal_cfg=None,
                 transform=None,
                 rnn_horizon=10):
        num_eps = data_file["data"].attrs["num_eps"]

        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.use_subgoal_eye_in_hand = use_subgoal_eye_in_hand
        self.transform = transform
        self.env_name = data_file["data"].attrs["env"]
        self.subtask_id = subtask_id

        self.policy_type = policy_type
        self.subgoal_cfg = subgoal_cfg
        
        subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
        
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()

        self.seq_length = rnn_horizon

        self.agentview_image_names = []
        self.frontview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.actions = []
        self.states = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.goal_images = []
        self.subgoal_indices = []


        self.proprios = []
        start_idx = 0 # Clip initial few steps of each episode
        self.total_len = 0

        if "image" in data_modality:
            count = 0
            for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()][start_idx:end_idx+1]
                eye_in_hand_image_names = data_file[f"data/ep_{i}/eye_in_hand_image_names"][()][start_idx:end_idx+1]
                self._seg_id_to_start_indices[seg_idx] = self.total_len
                self._seg_id_to_seg_length[seg_idx] = end_idx - start_idx + 1

                actions = data_file[f"data/ep_{i}/actions"][()][start_idx:end_idx+1]
                
                for j in range(len(agentview_image_names)):
                    self._idx_to_seg_id[self.total_len] = seg_idx
                    self.total_len += 1
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))
                    future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                    self.subgoal_indices.append(future_idx + count)
                    
                count = len(self.subgoal_indices)
                self.actions.append(actions)

            self.actions = np.vstack(self.actions)
            self.actions = safe_cuda(torch.from_numpy(self.actions))
            self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
            self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(max(self.subgoal_indices) == len(self.actions)-1)
                
        # else:
        #     for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):            
        #         low_dim_states = torch.from_numpy(data_file[f"data/ep_{i}/low_dim_states"][()][start_idx:end_idx+1])
        #         actions = data_file[f"data/ep_{i}/actions"][()][start_idx:end_idx+1]
        #         self.states.append(low_dim_states)
        #         self._seg_id_to_start_indices[seg_idx] = self.total_len
        #         self._seg_id_to_seg_length[seg_idx] = end_idx - start_idx + 1
        #         for j in range(len(actions)):
        #             self._idx_to_seg_id[self.total_len] = seg_idx
        #             self.total_len += 1
        #         self.actions.append(actions)

        #     self.states = safe_cuda(torch.from_numpy(np.vstack(self.states))).float()
        #     self.actions = safe_cuda(torch.from_numpy(np.vstack(self.actions))).float()
            
        print("Finish loading: ", self.total_len)

    @property
    def action_dim(self):
        return self.actions.shape[-1]


    @property
    def proprio_dim(self):
        if self.proprios == []:
            return 0
        else:
            return self.proprios.shape[-1]
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        seg_id = self._idx_to_seg_id[idx]
        seg_start_index = self._seg_id_to_start_indices[seg_id]
        seg_length = self._seg_id_to_seg_length[seg_id]

        index_in_seg = idx - seg_start_index
        end_index_in_seg = seg_length

        seq_begin_index = max(0, index_in_seg)
        seq_end_index = min(seg_length, index_in_seg + self.seq_length)
        padding = max(0, seq_begin_index + self.seq_length - seg_length)

        seq_begin_index += seg_start_index
        seq_end_index += seg_start_index
        
        action_seq = self.actions[seq_begin_index: seq_end_index].float()

        if "image" in self.data_modality:
            agentview_seq = self.agentview_images[seq_begin_index: seq_end_index]
            eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index: seq_end_index]
            subgoal_index = self.subgoal_indices[seq_end_index-1]
            if padding > 0:
                # Pad
                action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
                action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

                agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
                agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

                eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
                eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

            if self.use_eye_in_hand:
                obs_seq = torch.cat((agentview_seq, eye_in_hand_seq), dim=1).float()  / 255.
            else:
                obs_seq = agentview_seq.float() / 255.

            if self.use_subgoal_eye_in_hand:
                subgoal = torch.cat((self.agentview_images[subgoal_index],
                                     self.eye_in_hand_images[subgoal_index]), dim=1).float() / 255.
            else:
                subgoal = self.agentview_images[subgoal_index].float() / 255.
            return {"obs_seq": obs_seq,
                    "action": action_seq,
                    "subgoal": subgoal}

        # else:
        #     state_seq = self.states[seq_begin_index:seq_end_index]

        #     if padding > 0:
        #         action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
        #         action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

        #         state_end_pad = torch.repeat_interleave(state_seq[-1].unsqueeze(0), padding, dim=0)
        #         state_seq = torch.cat([state_seq] + [state_end_pad], dim=0)
            
        #     return {"obs": state_seq,
        #             "actions": action_seq}
        
        
class BaselineBCDataset(Dataset):
    def __init__(self,
                 data_file_name,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,                 
                 subgoal_cfg=None,
                 transform=None,
                 skill_training_cfg=None,                 
                 baseline_type="single_skill"):

        assert(baseline_type in ["single_skill", "gti", "rpl"])
        data_file = h5py.File(data_file_name, "r")

        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.transform = transform

        self.subgoal_cfg = subgoal_cfg
        self.skill_training_cfg = skill_training_cfg
        self.baseline_type = baseline_type

        self.use_subgoal_eye_in_hand = use_subgoal_eye_in_hand
        self.env_name = data_file["data"].attrs["env"]
        
        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.states = []
        self.actions = []
        self.proprios = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.goal_images = []
        self.subgoal_indices = []

        self.subgoal_transforms = transforms.Compose([
                     transforms.Resize((64, 64)),
                     transforms.Grayscale(num_output_channels=1)
                 ])


        self.num_eps = data_file["data"].attrs["num_eps"]

        for i in range(self.num_eps):
            actions = data_file[f"data/ep_{i}/actions"][()]
            for j in range(len(actions)):
                self.actions.append(actions[j])
        
        self.actions = np.array(self.actions)
        self.total_len = len(self.actions)
        self.actions = safe_cuda(torch.from_numpy(self.actions).float())

        # If GTI, also load goals / subgoals

        if self.baseline_type == "gti":
            self.subgoal_images = []
            count = 0
            for i in range(self.num_eps):
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()]
                for j in range(len(agentview_image_names)):
                    start_idx = j
                    future_idx = min(len(agentview_image_names)-1, j + subgoal_cfg["horizon"]) - start_idx
                    self.subgoal_indices.append(start_idx + future_idx)
            assert(len(self.actions) == len(self.subgoal_indices))

        if self.baseline_type == "rpl":
            self.subgoal_images = []
            count = 0
            for i in range(self.num_eps):
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()]
                for j in range(len(agentview_image_names)):
                    start_idx = j
                    future_idx = min(len(agentview_image_names)-1, j + subgoal_cfg["horizon"]) - start_idx
                    self.subgoal_indices.append(start_idx + future_idx)
            assert(len(self.actions) == len(self.subgoal_indices))
            
            
        if "image" in data_modality:
            for i in range(self.num_eps):
                agentview_image_names = data_file[f"data/ep_{i}/agentview_image_names"][()]
                if self.use_eye_in_hand:
                    eye_in_hand_image_names = data_file[f"data/ep_{i}/eye_in_hand_image_names"][()]
                for j in range(len(agentview_image_names)):
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    if self.use_eye_in_hand:
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))

            self.agentview_images =safe_cuda(torch.stack(self.agentview_images, dim=0))
            if self.use_eye_in_hand:
                self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))

            assert(len(self.actions) == len(self.agentview_images))

            
        if "proprio" in data_modality:
            gripper_states_list = []
            joint_states_list = []
            for i in range(self.num_eps):
                gripper_states = data_file[f"data/ep_{i}/gripper_states"][()]
                if self.skill_training_cfg.use_gripper:
                    for j in range(len(gripper_states)):
                        gripper_state = []
                        for k in range(j-5, j):
                            if k < 0:
                                gripper_state += gripper_states[0].tolist()
                            else:
                                gripper_state += gripper_states[k].tolist()
                        gripper_states_list.append(gripper_state)
                
                if self.skill_training_cfg.use_joints:
                    joint_states = torch.from_numpy(data_file[f"data/ep_{i}/joint_states"][()])
                    for j in range(len(joint_states)):
                        joint_states_list.append(joint_states[j])

            if self.skill_training_cfg.use_gripper and self.skill_training_cfg.use_joints:
                self.proprios = safe_cuda(torch.cat([torch.stack(joint_states_list, dim=0),
                                                     torch.tensor(gripper_states_list)], dim=1)).float()
            elif self.skill_training_cfg.use_gripper:
                self.proprios = safe_cuda(torch.tensor(gripper_states_list)).float()
            elif self.skill_training_cfg.use_joints:
                self.proprios = safe_cuda(torch.stack(joint_states_list, dim=0)).float()
            assert(len(self.proprios) == len(self.actions))
    @property
    def action_dim(self):
        return self.actions.shape[-1]

    @property
    def proprio_dim(self):
        if self.proprios == []:
            return 0
        else:
            return self.proprios.shape[-1]
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        action = self.actions[idx, ...]

        data = {"action": action}
        if "image" in self.data_modality:
            agentview_image = self.agentview_images[idx].float() / 255.
            if self.use_eye_in_hand:
                eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
                state_image = torch.cat((agentview_image, eye_in_hand_image), dim=0)
                if self.transform is not None:
                    state_image = self.transform(state_image)
                data["state_image"] = state_image
            else:
                if self.transform is not None:
                    state_image = self.transform(state_image)
                data["state_image"] = agentview_image
        if "proprio" in self.data_modality:
            data["proprio"] = self.proprios[idx]

        if self.baseline_type == "gti":
            if self.use_subgoal_eye_in_hand:
                subgoal_image = torch.cat((self.agentview_images[self.subgoal_indices[idx]].float() / 255.,
                                           self.eye_in_hand_images[self.subgoal_indices[idx]].float() / 255.))
            else:
                subgoal_image = self.agentview_images[self.subgoal_indices[idx]].float() / 255.
            
            data["subgoal"] = subgoal_image
            data["subgoal_target"] = self.subgoal_transforms(self.agentview_images[self.subgoal_indices[idx]].float() / 255.)

        elif self.baseline_type == "rpl":
            if self.use_subgoal_eye_in_hand:
                subgoal_image = torch.cat((self.agentview_images[self.subgoal_indices[idx]].float() / 255.,
                                           self.eye_in_hand_images[self.subgoal_indices[idx]].float() / 255.))
            else:
                subgoal_image = self.agentview_images[self.subgoal_indices[idx]].float() / 255.
            
            data["subgoal"] = subgoal_image
            
        return data
        

class VAEDataset(Dataset):
    def __init__(self,
                 data_file_name,
                 transform):
        data_file = h5py.File(data_file_name, "r")
        self.num_eps = data_file["data"].attrs["num_eps"]

        self.agentview_images = safe_cuda(torch.from_numpy((data_file["data/agentview_images"][()])))
        self.total_len = len(self.agentview_images)
        self.transform = transform
        print("Finished loading!")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return {"state": self.agentview_images[idx].float() / 255.,
                "target": self.transform(self.agentview_images[idx]).float() / 255.}


class MetaPolicyDataset(Dataset):
    def __init__(self,
                 data_file_name,
                 embedding_file_name,
                 subtasks_file_name,
                 use_eye_in_hand=False,
                 use_embedding=False,
                 seq_length=10,
                 transform=None):

        data_file = h5py.File(data_file_name, "r")
        embedding_file = h5py.File(embedding_file_name, "r")
        subtasks_file = h5py.File(subtasks_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.seq_length = seq_length
        self.transform = transform
        
        self.num_subtasks = subtasks_file["subtasks"].attrs["num_subtasks"]
        self.num_eps = subtasks_file["subtasks"].attrs["num_eps"]
        self.env_name = data_file["data"].attrs["env"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0

        for ep_idx in range(self.num_eps):
            try:
                saved_ep_subtasks_seq = subtasks_file["subtasks"][f"ep_subtasks_seq_{ep_idx}"][()]
            except:
                continue
            for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                if k == len(saved_ep_subtasks_seq) - 1:
                    e_idx = end_idx + 1
                else:
                    e_idx = end_idx
                agentview_image_names = data_file[f"data/ep_{ep_idx}/agentview_image_names"][()][start_idx:e_idx]
                eye_in_hand_image_names = data_file[f"data/ep_{ep_idx}/eye_in_hand_image_names"][()][start_idx:e_idx]

                embeddings = embedding_file[f"data/ep_{ep_idx}/embedding"][()][start_idx:e_idx]
                for j in range(len(agentview_image_names)):
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))
                    self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                    
                    self.subtask_labels.append(subtask_label)
                    self.total_len += 1

        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
                    
        self.agentview_images =safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("Subtask: ", self.subtask_labels.shape)
        data_file.close()
        embedding_file.close()
        subtasks_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        agentview_image = self.agentview_images[idx].float() / 255.
        if self.use_eye_in_hand:
            eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
            state_image = torch.cat([agentview_image, eye_in_hand_image], dim=0)
        else:
            state_image = agentview_image
        subgoal_embedding = self.subgoal_embeddings[idx].float()
        subtask_label = self.subtask_labels[idx]
        return {"state_image": state_image, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks)}, {"embedding": subgoal_embedding, "id": subtask_label}



class MultitaskMetaPolicyDataset(Dataset):
    def __init__(self,
                 data_file_name,
                 embedding_file_name,
                 subtasks_file_name,
                 task_id,
                 use_eye_in_hand=False,
                 use_embedding=False,
                 seq_length=10,
                 testing_percentage=1.0,
                 training_task_id=-1,
                 transform=None):

        data_file = h5py.File(data_file_name, "r")
        embedding_file = h5py.File(embedding_file_name, "r")
        subtasks_file = h5py.File(subtasks_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.seq_length = seq_length
        self.transform = transform
        
        self.num_subtasks = subtasks_file["subtasks"].attrs["num_subtasks"]

        self.task_id = task_id
        self.training_task_id = training_task_id

        if training_task_id == -1:
            self.ep_indices = data_file["data/task"].attrs[f"{self.task_id}"]
        else:
            ep_indices = []
            ids = [[0, 4], [2, 3], [1, 7]][self.training_task_id]

            for i in ids:
                ep_indices += (data_file["data/task"].attrs[f"{i}"]).tolist()
            self.ep_indices = ep_indices
                
        if testing_percentage < 1.0:
            self.ep_indices = self.ep_indices[:int(len(self.ep_indices) * testing_percentage)]
        self.num_eps = len(self.ep_indices)
        print("Number of eps: ", self.num_eps)
        self.env_name = data_file["data"].attrs["env"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0

        for ep_idx in self.ep_indices:
            try:
                saved_ep_subtasks_seq = subtasks_file["subtasks"][f"ep_subtasks_seq_{ep_idx}"][()]
            except:
                continue
            for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                if k == len(saved_ep_subtasks_seq) - 1:
                    e_idx = end_idx + 1
                else:
                    e_idx = end_idx
                agentview_image_names = data_file[f"data/ep_{ep_idx}/agentview_image_names"][()][start_idx:e_idx]
                eye_in_hand_image_names = data_file[f"data/ep_{ep_idx}/eye_in_hand_image_names"][()][start_idx:e_idx]

                embeddings = embedding_file[f"data/ep_{ep_idx}/embedding"][()][start_idx:e_idx]
                for j in range(len(agentview_image_names)):
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))
                    self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                    
                    self.subtask_labels.append(subtask_label)
                    self.total_len += 1

        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
                    
        self.agentview_images =safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("Subtask: ", self.subtask_labels.shape)
        data_file.close()
        embedding_file.close()
        subtasks_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        agentview_image = self.agentview_images[idx].float() / 255.
        if self.use_eye_in_hand:
            eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
            state_image = torch.cat([agentview_image, eye_in_hand_image], dim=0)
        else:
            state_image = agentview_image
        subgoal_embedding = self.subgoal_embeddings[idx].float()
        subtask_label = self.subtask_labels[idx]
        return {"state_image": state_image, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks)}, {"embedding": subgoal_embedding, "id": subtask_label}
    
    

class MetaRNNPolicyDataset(Dataset):
    def __init__(self,
                 data_file_name,
                 embedding_file_name,
                 subtasks_file_name,
                 use_eye_in_hand=False,
                 use_embedding=False,
                 seq_length=10,
                 transform=None):

        data_file = h5py.File(data_file_name, "r")
        embedding_file = h5py.File(embedding_file_name, "r")
        self.embedding_dim = 16 # embedding_file["data"].attrs["embedding_dim"]
        subtasks_file = h5py.File(subtasks_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.seq_length = seq_length
        self.transform = transform
        
        self.num_subtasks = subtasks_file["subtasks"].attrs["num_subtasks"]
        self.num_eps = subtasks_file["subtasks"].attrs["num_eps"]
        self.env_name = data_file["data"].attrs["env"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.subtask_labels = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.goal_images = []

        self.total_len = 0
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()
        seg_idx = 0
        
        for ep_idx in range(self.num_eps):
            try:
                saved_ep_subtasks_seq = subtasks_file["subtasks"][f"ep_subtasks_seq_{ep_idx}"][()]
            except:
                continue
            for (start_idx, end_idx, subtask_label) in saved_ep_subtasks_seq:
                self._seg_id_to_start_indices[seg_idx] = self.total_len
                self._seg_id_to_seg_length[seg_idx] = end_idx - start_idx + 1

                goal_image_name = data_file[f"data/ep_{ep_idx}/agentview_image_names"][()][-1]
                agentview_image_names = data_file[f"data/ep_{ep_idx}/agentview_image_names"][()][start_idx:end_idx+1]
                eye_in_hand_image_names = data_file[f"data/ep_{ep_idx}/eye_in_hand_image_names"][()][start_idx:end_idx+1]        
                for j in range(len(agentview_image_names)):
                    self.agentview_images.append(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)))
                    self.eye_in_hand_images.append(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)))

                    self.goal_images.append(torch.from_numpy(np.array(Image.open(goal_image_name)).transpose(2, 0, 1)))
                    self.subtask_labels.append(subtask_label)

                    self._idx_to_seg_id[self.total_len] = seg_idx
                    self.total_len += 1
                seg_idx += 1

        self.agentview_images =safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.goal_images = safe_cuda(torch.stack(self.goal_images, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("Subtask: ", self.subtask_labels.shape)
        data_file.close()
        embedding_file.close()
        subtasks_file.close()


    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        seg_id = self._idx_to_seg_id[idx]
        seg_start_index = self._seg_id_to_start_indices[seg_id]
        seg_length = self._seg_id_to_seg_length[seg_id]

        index_in_seg = idx - seg_start_index
        end_index_in_seg = seg_length

        seq_begin_index = max(0, index_in_seg)
        seq_end_index = min(seg_length, index_in_seg + self.seq_length)
        padding = max(0, seq_begin_index + self.seq_length - seg_length)

        seq_begin_index += seg_start_index
        seq_end_index += seg_start_index

        agentview_seq = self.agentview_images[seq_begin_index:seq_end_index]
        eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index:seq_end_index]
        goal_seq = self.goal_images[seq_begin_index:seq_end_index]
        subtask_label_seq = self.subtask_labels[seq_begin_index:seq_end_index]

        if padding > 0:
            agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
            agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

            eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
            eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

            goal_end_pad = torch.repeat_interleave(goal_seq[-1].unsqueeze(0), padding, dim=0)
            goal_seq = torch.cat([goal_seq] + [goal_end_pad], dim=0)

            subtask_label_end_pad = torch.repeat_interleave(subtask_label_seq[-1].unsqueeze(0), padding, dim=0)
            subtask_label_seq = torch.cat([subtask_label_seq] + [subtask_label_end_pad], dim=0)

        if self.use_eye_in_hand:
            state_seq = torch.cat((agentview_seq, eye_in_hand_seq), dim=1).float() / 255.
        else:
            state_seq = agentview_seq.float() / 255.
        goal_seq = goal_seq.float() / 255.
        if self.transform is not None:
            goal_seq = self.transform(goal_seq)
        return {"state": state_seq,
                "goal": goal_seq}, subtask_label_seq

