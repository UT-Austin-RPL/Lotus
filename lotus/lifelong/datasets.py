import copy

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from robomimic.utils.file_utils import create_hdf5_filter_key
from torch.utils.data import Dataset
import h5py

"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""
new_task_demo_num = 0
old_task_demo_num = 1

def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    new_task_name="@default@",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key

    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)



## Skill Learning Dataset
from ..skill_learning.models.model_utils import *
from ..skill_learning.models.torch_utils import to_onehot
class SubtaskSequenceDataset(Dataset):
    def __init__(self,
                 data_file_list,
                 subtask_file_list,
                 subtask_id,
                 data_modality=["image", "proprio"],             
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 subgoal_cfg=None,
                 seq_len=10,
                 task_embs=None,
                 goal_modality="BUDS",
                 dinov2_file_list=[],
                 new_task_name="@default@",
                 demo_range=range(0, 50)):
        # demo_num = data_file["data"].attrs["demo_num"]
        self.dataset_num = len(data_file_list)
        self.data_modality = data_modality
        self.goal_modality = goal_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.use_subgoal_eye_in_hand = use_subgoal_eye_in_hand
        self.subtask_id = subtask_id

        self.subgoal_cfg = subgoal_cfg
        
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()

        self.seq_length = seq_len

        self.agentview_image_names = []
        self.frontview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.actions = []
        self.states = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.gripper_states = []
        self.joint_states = []
        self.ee_states = []
        self.goal_images = []
        self.dinov2_features = []
        self.subgoal_indices = []


        self.proprios = []
        start_idx = 0 # Clip initial few steps of each episode
        self.total_len = 0
        count = 0
        self.not_use_this_dataset = False

        if not dinov2_file_list:
            for file_id, (data_file, subtask_file) in enumerate(zip(data_file_list, subtask_file_list)):
                subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
                for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):
                    if isinstance(new_task_name, list):
                        if any(name in data_file.filename for name in new_task_name):
                            demo_range = range(0, new_task_demo_num)
                        else:
                            demo_range = range(0, old_task_demo_num)
                    else:
                        if new_task_name in data_file.filename:
                            demo_range = range(0, new_task_demo_num)# range(30, 50)
                        else:
                            demo_range = range(0, old_task_demo_num)# range(40, 50)

                    if i not in demo_range:
                        continue
                    agentview_images = data_file[f"data/demo_{i}/obs/agentview_rgb"][()][start_idx:end_idx+1]
                    eye_in_hand_images = data_file[f"data/demo_{i}/obs/eye_in_hand_rgb"][()][start_idx:end_idx+1]

                    self._seg_id_to_start_indices[(file_id, seg_idx)] = self.total_len
                    self._seg_id_to_seg_length[(file_id, seg_idx)] = end_idx - start_idx + 1

                    actions = data_file[f"data/demo_{i}/actions"][()][start_idx:end_idx+1]
                    gripper_states = data_file[f"data/demo_{i}/obs/gripper_states"][()][start_idx:end_idx+1]
                    joint_states = data_file[f"data/demo_{i}/obs/joint_states"][()][start_idx:end_idx+1]
                    ee_states = data_file[f"data/demo_{i}/obs/ee_states"][()][start_idx:end_idx+1]
                    
                    for j in range(end_idx - start_idx + 1):
                        self._idx_to_seg_id[self.total_len] = (file_id, seg_idx)
                        self.total_len += 1
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                        self.subgoal_indices.append(future_idx + count)
                        
                    count = len(self.subgoal_indices)
                    self.actions.append(actions)
                    self.gripper_states.append(gripper_states)
                    self.joint_states.append(joint_states)
                    self.ee_states.append(ee_states)

            if len(self.actions) == 0:
                self.not_use_this_dataset = True
                return None
            self.actions = np.vstack(self.actions)
            self.actions = safe_cuda(torch.from_numpy(self.actions))
            self.gripper_states = np.vstack(self.gripper_states)
            self.gripper_states = safe_cuda(torch.from_numpy(self.gripper_states))
            self.joint_states = np.vstack(self.joint_states)
            self.joint_states = safe_cuda(torch.from_numpy(self.joint_states))
            self.ee_states = np.vstack(self.ee_states)
            self.ee_states = safe_cuda(torch.from_numpy(self.ee_states))
            self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
            self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(max(self.subgoal_indices) == len(self.actions)-1)
        
        else: # use dinov2 features
            for file_id, (data_file, subtask_file, dinov2_file) in enumerate(zip(data_file_list, subtask_file_list, dinov2_file_list)):
                subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
                for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):
                    if isinstance(new_task_name, list):
                        if any(name in data_file.filename for name in new_task_name):
                            demo_range = range(0, new_task_demo_num)
                        else:
                            demo_range = range(0, old_task_demo_num)
                    else:
                        if new_task_name in data_file.filename:
                            demo_range = range(0, new_task_demo_num)# range(30, 50)
                        else:
                            demo_range = range(0, old_task_demo_num)# range(40, 50)
                    
                    if i not in demo_range:
                        continue
                    agentview_images = data_file[f"data/demo_{i}/obs/agentview_rgb"][()][start_idx:end_idx+1]
                    eye_in_hand_images = data_file[f"data/demo_{i}/obs/eye_in_hand_rgb"][()][start_idx:end_idx+1]
                    dinov2_features = dinov2_file[f"data/demo_{i}/embedding"][()][start_idx:end_idx+1]

                    self._seg_id_to_start_indices[(file_id, seg_idx)] = self.total_len
                    self._seg_id_to_seg_length[(file_id, seg_idx)] = end_idx - start_idx + 1

                    actions = data_file[f"data/demo_{i}/actions"][()][start_idx:end_idx+1]
                    gripper_states = data_file[f"data/demo_{i}/obs/gripper_states"][()][start_idx:end_idx+1]
                    joint_states = data_file[f"data/demo_{i}/obs/joint_states"][()][start_idx:end_idx+1]
                    ee_states = data_file[f"data/demo_{i}/obs/ee_states"][()][start_idx:end_idx+1]
                    
                    for j in range(end_idx - start_idx + 1):
                        self._idx_to_seg_id[self.total_len] = (file_id, seg_idx)
                        self.total_len += 1
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                        self.subgoal_indices.append(future_idx + count)
                        
                    count = len(self.subgoal_indices)
                    self.actions.append(actions)
                    self.gripper_states.append(gripper_states)
                    self.joint_states.append(joint_states)
                    self.ee_states.append(ee_states)
                    self.dinov2_features.append(dinov2_features)

            if len(self.actions) == 0:
                self.not_use_this_dataset = True
                return None
            self.actions = np.vstack(self.actions)
            self.actions = safe_cuda(torch.from_numpy(self.actions))
            self.gripper_states = np.vstack(self.gripper_states)
            self.gripper_states = safe_cuda(torch.from_numpy(self.gripper_states))
            self.joint_states = np.vstack(self.joint_states)
            self.joint_states = safe_cuda(torch.from_numpy(self.joint_states))
            self.ee_states = np.vstack(self.ee_states)
            self.ee_states = safe_cuda(torch.from_numpy(self.ee_states))
            self.dinov2_features = np.vstack(self.dinov2_features)
            self.dinov2_features = safe_cuda(torch.from_numpy(self.dinov2_features))
            self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
            self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(len(self.dinov2_features) == len(self.actions))
            assert(max(self.subgoal_indices) == len(self.actions)-1)
            
        print(f"Finish loading subtask_{subtask_id}: ", self.total_len)

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
        data={}
        data["obs"]={}
        file_id, seg_id = self._idx_to_seg_id[idx]
        seg_start_index = self._seg_id_to_start_indices[(file_id, seg_id)]
        seg_length = self._seg_id_to_seg_length[(file_id, seg_id)]

        index_in_seg = idx - seg_start_index
        end_index_in_seg = seg_length

        seq_begin_index = max(0, index_in_seg)
        seq_end_index = min(seg_length, index_in_seg + self.seq_length)
        padding = max(0, seq_begin_index + self.seq_length - seg_length)

        seq_begin_index += seg_start_index
        seq_end_index += seg_start_index
        
        action_seq = self.actions[seq_begin_index: seq_end_index].float()
        gripper_state_seq = self.gripper_states[seq_begin_index: seq_end_index].float()
        joint_state_seq = self.joint_states[seq_begin_index: seq_end_index].float()
        ee_state_seq = self.ee_states[seq_begin_index: seq_end_index].float()

        if self.goal_modality != "dinov2":
            agentview_seq = self.agentview_images[seq_begin_index: seq_end_index]
            eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index: seq_end_index]
            # use single subgoal or seq subgoals in the sequence
            subgoal_index = self.subgoal_indices[seq_end_index-1] #TODO: need to reconsider this
            subgoal_seq = self.subgoal_indices[seq_begin_index: seq_end_index]
            if padding > 0:
                # Pad
                action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
                action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

                gripper_state_end_pad = torch.repeat_interleave(gripper_state_seq[-1].unsqueeze(0), padding, dim=0)
                gripper_state_seq = torch.cat([gripper_state_seq] + [gripper_state_end_pad], dim=0)

                joint_state_end_pad = torch.repeat_interleave(joint_state_seq[-1].unsqueeze(0), padding, dim=0)
                joint_state_seq = torch.cat([joint_state_seq] + [joint_state_end_pad], dim=0)

                ee_state_end_pad = torch.repeat_interleave(ee_state_seq[-1].unsqueeze(0), padding, dim=0)
                ee_state_seq = torch.cat([ee_state_seq] + [ee_state_end_pad], dim=0)

                agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
                agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

                eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
                eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

                subgoal_end_pad = [subgoal_seq[-1]] * padding
                subgoal_seq = subgoal_seq + subgoal_end_pad


            if self.use_eye_in_hand:
                agentview_rgb = agentview_seq.float() / 255.
                eye_in_hand_rgb = eye_in_hand_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb
                data["obs"]["eye_in_hand_rgb"] = eye_in_hand_rgb    
            else:
                agentview_rgb = agentview_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb

            if self.goal_modality == "BUDS":
                if self.use_subgoal_eye_in_hand:
                    # TODO:need to update
                    subgoal = torch.cat((self.agentview_images[subgoal_index],
                                        self.eye_in_hand_images[subgoal_index]), dim=1).float() / 255.
                    data["obs"]["subgoal"] = subgoal
                else:
                    # # use individual subgoal in the sequence
                    # subgoal = [self.agentview_images[i] for i in subgoal_seq]
                    # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float() / 255.

                    # repeat final subgoal in the sequence
                    subgoal = self.agentview_images[subgoal_index].float() / 255.
                    data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)

            elif self.goal_modality == "ee_states":
                # # use individual subgoal in the sequence
                # subgoal = [torch.cat([self.ee_states[i]] + [self.gripper_states[i]], dim=0) for i in subgoal_seq]
                # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

                # repeat final subgoal in the sequence
                subgoal = torch.cat([self.ee_states[subgoal_index]] + [self.gripper_states[subgoal_index]], dim=0)
                data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1)

            elif self.goal_modality == "joint_states":
                # # use individual subgoal in the sequence
                # subgoal = [torch.cat([self.joint_states[i]] + [self.gripper_states[i]], dim=0) for i in subgoal_seq]
                # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

                # repeat final subgoal in the sequence
                subgoal = torch.cat([self.joint_states[subgoal_index]] + [self.gripper_states[subgoal_index]], dim=0)
                data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1)

            data["actions"] = action_seq
            data['obs']["gripper_states"] = gripper_state_seq
            data['obs']["joint_states"] = joint_state_seq
            return data

        else: # use dinov2 features
            dinov2_feature_seq = self.dinov2_features[seq_begin_index: seq_end_index].float()
            agentview_seq = self.agentview_images[seq_begin_index: seq_end_index]
            eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index: seq_end_index]
            # use single subgoal or seq subgoals in the sequence
            subgoal_index = self.subgoal_indices[seq_end_index-1] #TODO: need to reconsider this
            subgoal_seq = self.subgoal_indices[seq_begin_index: seq_end_index]
            if padding > 0:
                # Pad
                action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
                action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

                gripper_state_end_pad = torch.repeat_interleave(gripper_state_seq[-1].unsqueeze(0), padding, dim=0)
                gripper_state_seq = torch.cat([gripper_state_seq] + [gripper_state_end_pad], dim=0)

                joint_state_end_pad = torch.repeat_interleave(joint_state_seq[-1].unsqueeze(0), padding, dim=0)
                joint_state_seq = torch.cat([joint_state_seq] + [joint_state_end_pad], dim=0)

                ee_state_end_pad = torch.repeat_interleave(ee_state_seq[-1].unsqueeze(0), padding, dim=0)
                ee_state_seq = torch.cat([ee_state_seq] + [ee_state_end_pad], dim=0)

                agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
                agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

                eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
                eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

                dinov2_feature_end_pad = torch.repeat_interleave(dinov2_feature_seq[-1].unsqueeze(0), padding, dim=0)
                dinov2_feature_seq = torch.cat([dinov2_feature_seq] + [dinov2_feature_end_pad], dim=0)

                subgoal_end_pad = [subgoal_seq[-1]] * padding
                subgoal_seq = subgoal_seq + subgoal_end_pad


            if self.use_eye_in_hand:
                agentview_rgb = agentview_seq.float() / 255.
                eye_in_hand_rgb = eye_in_hand_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb
                data["obs"]["eye_in_hand_rgb"] = eye_in_hand_rgb    
            else:
                agentview_rgb = agentview_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb

            if self.goal_modality == "dinov2":
                # # use individual subgoal in the sequence
                # subgoal = [self.dinov2_features[i] for i in subgoal_seq]
                # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

                # repeat final subgoal in the sequence
                subgoal = self.dinov2_features[subgoal_index].float()
                data["obs"]["subgoal"] = subgoal.repeat(self.seq_length, 1)
            else:
                pass ## TODO

            data["actions"] = action_seq
            data['obs']["gripper_states"] = gripper_state_seq
            data['obs']["joint_states"] = joint_state_seq
            return data


class SkillLearningDataset():
    def __init__(self,
                 data_file_name_list,
                 subtasks_file_name_list,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 subgoal_cfg=None,
                 subtask_id=[],
                 seq_len=10,
                 task_embs=None,
                 goal_modality="BUDS",
                 new_task_name="@default@",
                 demo_range=range(0, 50),
                 used_data_file_name_list=[],):
    
        self.f_list = []
        self.dinov2_f_list = []
        self.train_dataset_id = []
        self.new_task_name = new_task_name
        self.demo_range = demo_range
        self.goal_modality = goal_modality
        for data_file_name in data_file_name_list:
            if any(used_data_file_name in data_file_name for used_data_file_name in used_data_file_name_list):
                self.f_list.append(h5py.File(data_file_name, "r"))
                if goal_modality == "dinov2":
                    import re
                    dinov2_feature_file_name = re.sub(r"(datasets/)([^/]+)(/)", r"\1dinov2/\2\3", data_file_name)
                    self.dinov2_f_list.append(h5py.File(dinov2_feature_file_name, "r"))

        self.subtasks_f_list = []
        for subtasks_file_name in subtasks_file_name_list:
            if any(used_data_file_name in subtasks_file_name for used_data_file_name in used_data_file_name_list):
                self.subtasks_f_list.append(h5py.File(subtasks_file_name, "r"))

        self.subtask_id = subtask_id
        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.num_subtasks = self.subtasks_f_list[0]["subtasks"].attrs["num_subtasks"]
        self.subgoal_cfg = subgoal_cfg
        self.seq_len = seq_len
        self.task_embs = task_embs

        for subtasks_f in self.subtasks_f_list:
            print("subtasks distance score:",subtasks_f["subtasks"].attrs["score"])
            if isinstance(new_task_name, list):
                for task in new_task_name:
                    if task == "@default@":
                        self.train_dataset_id = list(range(self.num_subtasks))
                    if task in subtasks_f.filename:
                        for key in subtasks_f['subtasks']:
                            if 'segmentation' in subtasks_f['subtasks'][key]:
                                data = subtasks_f['subtasks'][key]['segmentation'][()]
                                if data.size != 0:
                                    x = key.split('_')[-1]
                                    self.train_dataset_id.append(int(x))  
            else:
                if new_task_name == "@default@":
                    self.train_dataset_id = list(range(self.num_subtasks))
                if new_task_name in subtasks_f.filename:
                    for key in subtasks_f['subtasks']:
                        if 'segmentation' in subtasks_f['subtasks'][key]:
                            data = subtasks_f['subtasks'][key]['segmentation'][()]
                            if data.size != 0:
                                x = key.split('_')[-1]
                                self.train_dataset_id.append(int(x))
            self.train_dataset_id = sorted(list(set(self.train_dataset_id)))
        print("train_dataset_id:", self.train_dataset_id)
        self.datasets = []

    def get_dataset(self, idx):
        if self.subtask_id != []:
            if idx not in self.subtask_id:
                return None

        dataset = SubtaskSequenceDataset(self.f_list,
                                        self.subtasks_f_list,
                                        idx,
                                        data_modality=self.data_modality,
                                        use_eye_in_hand=self.use_eye_in_hand,
                                        use_subgoal_eye_in_hand=self.subgoal_cfg.use_eye_in_hand,
                                        subgoal_cfg=self.subgoal_cfg,
                                        seq_len=self.seq_len,
                                        task_embs=self.task_embs,
                                        goal_modality=self.goal_modality,
                                        dinov2_file_list=self.dinov2_f_list,
                                        new_task_name=self.new_task_name,
                                        demo_range=self.demo_range,)
        if dataset.not_use_this_dataset:
            return None
        # print(idx, len(dataset))
        return dataset
    
    def close(self):
        for f in self.f_list:
            f.close()
        for subtasks_f in self.subtasks_f_list:
            self.subtasks_f.close()



class MetaPolicyDataset(Dataset):
    def __init__(self,
                 data_file_name_list,
                 embedding_file_name,
                 subtasks_file_name_list,
                 task_names,
                 task_embs,
                 use_eye_in_hand=False,
                 ):

        embedding_file = h5py.File(embedding_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.num_subtasks = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["num_subtasks"]
        self.demo_num = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["demo_num"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []
        self.task_idx = []
        self.task_embs = task_embs

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0

        for data_file_name, subtasks_file_name in zip(data_file_name_list, subtasks_file_name_list):
            data_file = h5py.File(data_file_name, "r")
            dataset_category, dataset_name = data_file_name.split("/")[1:]
            dataset_name = dataset_name.split(".")[0]
            subtasks_file = h5py.File(subtasks_file_name, "r")
            task_idx = task_names.index(dataset_name.split("_demo")[0])

            for ep_idx in range(self.demo_num):
                try:
                    saved_ep_subtasks_seq = subtasks_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
                except:
                    continue
                for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                    if k == len(saved_ep_subtasks_seq) - 1:
                        e_idx = end_idx + 1
                    else:
                        e_idx = end_idx
                    agentview_images = data_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx:e_idx]
                    eye_in_hand_images = data_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()][start_idx:e_idx]

                    embeddings = embedding_file[f"{dataset_name}/demo_{ep_idx}/embedding"][()][start_idx:e_idx]
                    for j in range(len(agentview_images)):
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                        
                        self.subtask_labels.append(subtask_label)
                        self.task_idx.append(task_idx)
                        self.total_len += 1
            
            data_file.close()
        embedding_file.close()


        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
         
        self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("MetaPolicyDataset: ", self.subtask_labels.shape)
        embedding_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        agentview_image = self.agentview_images[idx].float() / 255.
        if self.use_eye_in_hand:
            eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
        #     state_image = torch.cat([agentview_image, eye_in_hand_image], dim=0)
        # else:
        #     state_image = agentview_image
        subgoal_embedding = self.subgoal_embeddings[idx].float()
        subtask_label = self.subtask_labels[idx]
        task_idx = self.task_idx[idx]
        task_emb = self.task_embs[task_idx]
        # return {"state_image": state_image, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks)},{"embedding": subgoal_embedding, "id": subtask_label}
        data = {}
        data["obs"] = {"agentview_rgb": agentview_image, "task_emb": task_emb, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks), "id": subtask_label}
        return data


class MetaPolicySequenceDataset(Dataset):
    def __init__(self,
                 data_file_name_list,
                 embedding_file_name,
                 subtasks_file_name_list,
                 task_names,
                 task_embs,
                 use_eye_in_hand=False,
                 seq_length=10,
                 new_task_name="@default@",
                 demo_range=range(0, 50),
                 used_data_file_name_list=[],
                 ):

        embedding_file = h5py.File(embedding_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.seq_length = seq_length
        self.num_subtasks = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["num_subtasks"]
        self.demo_num = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["demo_num"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []
        self.task_idx = []
        self.task_embs = task_embs

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()
        seg_idx = 0

        # demo_num_10_file_names =[
        #     "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        #     "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        #     "pick_up_the_salad_dressing_and_place_it_in_the_basket",

        #     "put_the_bowl_on_the_stove",
        #     "put_the_cream_cheese_in_the_bowl",
        # ]

        for data_file_name, subtasks_file_name in zip(data_file_name_list, subtasks_file_name_list):
            if not any(used_data_file_name in data_file_name for used_data_file_name in used_data_file_name_list):
                continue
            data_file = h5py.File(data_file_name, "r")
            dataset_category, dataset_name = data_file_name.split("/")[1:]
            dataset_name = dataset_name.split(".")[0]
            subtasks_file = h5py.File(subtasks_file_name, "r")
            task_idx = task_names.index(dataset_name.split("_demo")[0])

            if isinstance(new_task_name, list):
                if any(name in data_file.filename for name in new_task_name):
                    demo_range = range(0, new_task_demo_num)
                else:
                    demo_range = range(0, old_task_demo_num)
            else:
                if new_task_name in data_file.filename:
                    demo_range = range(0, new_task_demo_num)# range(30, 50)
                else:
                    demo_range = range(0, old_task_demo_num)# range(40, 50)

            for ep_idx in range(self.demo_num):
                if ep_idx not in demo_range:
                    continue
                try:
                    saved_ep_subtasks_seq = subtasks_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
                except:
                    continue
                for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                    if k == len(saved_ep_subtasks_seq) - 1:
                        e_idx = end_idx + 1
                    else:
                        e_idx = end_idx
                    self._seg_id_to_start_indices[seg_idx] = self.total_len
                    self._seg_id_to_seg_length[seg_idx] = end_idx - start_idx + 1
                    agentview_images = data_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx:e_idx]
                    eye_in_hand_images = data_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()][start_idx:e_idx]

                    embeddings = embedding_file[f"{dataset_name}/demo_{ep_idx}/embedding"][()][start_idx:e_idx]
                    for j in range(len(agentview_images)):
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                        
                        self.subtask_labels.append(subtask_label)
                        self.task_idx.append(task_idx)
                        self._idx_to_seg_id[self.total_len] = seg_idx
                        self.total_len += 1
                    seg_idx += 1
            
            data_file.close()
        embedding_file.close()


        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
         
        self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("MetaPolicyDataset: ", self.subtask_labels.shape)
        embedding_file.close()

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
        subgoal_embedding_seq = self.subgoal_embeddings[seq_begin_index:seq_end_index]
        subtask_label_seq = self.subtask_labels[seq_begin_index:seq_end_index]
        task_idx_seq = self.task_idx[seq_begin_index:seq_end_index]

        if padding > 0:
            agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
            agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

            eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
            eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

            subgoal_embedding_end_pad = torch.repeat_interleave(subgoal_embedding_seq[-1].unsqueeze(0), padding, dim=0)
            subgoal_embedding_seq = torch.cat([subgoal_embedding_seq] + [subgoal_embedding_end_pad], dim=0)

            subtask_label_end_pad = torch.repeat_interleave(subtask_label_seq[-1].unsqueeze(0), padding, dim=0)
            subtask_label_seq = torch.cat([subtask_label_seq] + [subtask_label_end_pad], dim=0)

            task_idx_end_pad = [task_idx_seq[-1]] * padding
            task_idx_seq.extend(task_idx_end_pad)

        
        agentview_seq = agentview_seq.float() / 255.
        subgoal_embedding_seq = subgoal_embedding_seq.float()
        task_emb_seq = [self.task_embs[task_idx] for task_idx in task_idx_seq]

        data = {}
        data["task_emb"] = task_emb_seq[0]
        data["obs"] = {"agentview_rgb": agentview_seq, "embedding": subgoal_embedding_seq, "id_vector": to_onehot(subtask_label_seq, self.num_subtasks), "id": subtask_label_seq}
        return data