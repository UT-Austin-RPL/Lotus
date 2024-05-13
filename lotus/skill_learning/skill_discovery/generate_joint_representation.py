import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import numpy as np
import cv2


import init_path
from models.model_utils import safe_cuda, Modality_input, SensorFusion
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fusion-dataset-name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["agentview", "eye_in_hand", "force"]
    )

    
    parser.add_argument(
        '--alpha-kl',
        type=float,
        default=0.05,
    )

    parser.add_argument(
        '--z-dim',
        type=int,
        default=128,
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    modalities = args.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{args.alpha_kl}"    
    if args.checkpoint:
        modality_str += "_checkpoint"
    if args.no_skip:
        modality_str += "_no_skip"
    sensor_fusion = safe_cuda(SensorFusion(z_dim=args.z_dim, use_skip_connection=not args.no_skip, modalities=modalities))

    if args.fusion_dataset_name == "":
        args.fusion_dataset_name = args.dataset_name
    sensor_fusion.load_state_dict(torch.load(f"results/{args.fusion_dataset_name}/Fusion_{modality_str}_{args.z_dim}.pth"))
    out_parent_dir = f"datasets/{dataset_name}"
    hdf5_path = os.path.join(out_parent_dir, "demo.hdf5")

    h5py_f = h5py.File(hdf5_path, "r")

    num_eps = h5py_f["data"].attrs["num_eps"]

    save_hdf5_path = os.path.join(out_parent_dir, f"embedding_{modality_str}_{args.z_dim}.hdf5")
    save_h5py_f = h5py.File(save_hdf5_path, "w")

    grp = save_h5py_f.create_group("data")
    
    for i in range(num_eps):
        ft_states = h5py_f[f"data/ep_{i}/ft_states"][()]
        proprio_states = h5py_f[f"data/ep_{i}/proprio_states"][()]
        agentview_image_names = h5py_f[f"data/ep_{i}/agentview_image_names"][()]
        eye_in_hand_image_names = h5py_f[f"data/ep_{i}/eye_in_hand_image_names"][()]

        embeddings = []
        for j in range(len(ft_states)):
            ft_state = safe_cuda(torch.from_numpy(ft_states[j, ...]).float().unsqueeze(0))
            proprio_state = safe_cuda(torch.from_numpy(proprio_states[j, ...]).float().unsqueeze(0))
            agentview_image = safe_cuda(torch.from_numpy(np.array(Image.open(agentview_image_names[j])).transpose(2, 0, 1)).float().unsqueeze(0)) / 255.
            eye_in_hand_image = safe_cuda(torch.from_numpy(np.array(Image.open(eye_in_hand_image_names[j])).transpose(2, 0, 1)).float().unsqueeze(0)) / 255.
            x = Modality_input(frontview=None, agentview=agentview_image, eye_in_hand=eye_in_hand_image, force=ft_state, proprio=proprio_state)
            embedding = sensor_fusion(x, encoder_only=True).cpu().detach().numpy()
            embeddings.append(embedding)

        assert(len(embeddings)==len(agentview_image_names))
        ep_data_grp = grp.create_group(f"ep_{i}")
        ep_data_grp.create_dataset("embedding", data=np.stack(embeddings, axis=0))

        ep_data_grp.attrs['len'] = len(ft_states)
    save_h5py_f.close()
    h5py_f.close()
            
if __name__ == "__main__":
    main()
