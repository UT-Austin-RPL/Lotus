import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import numpy as np
import cv2

import os
import argparse
import init_path
from models.model_utils import safe_cuda, Modality_input, SensorFusion


# Dataset_Name_List = [
#     "../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo",
#     "../datasets/libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo",
# ]

Dataset_Name_List = [
    "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
    "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
    "../datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    "../datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
    "../datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
    "../datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
    "../datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
    "../datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
    "../datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
    "../datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
]

# Dataset_Name_List = [
#     "../datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo",
# ]

# Dataset_Name_List = [
#     "../datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo",
#     "../datasets/libero_goal/put_the_bowl_on_the_stove_demo",
#     "../datasets/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo",
#     "../datasets/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo",
#     "../datasets/libero_goal/put_the_bowl_on_top_of_the_cabinet_demo",
#     "../datasets/libero_goal/push_the_plate_to_the_front_of_the_stove_demo",
#     "../datasets/libero_goal/put_the_cream_cheese_in_the_bowl_demo",
#     "../datasets/libero_goal/turn_on_the_stove_demo",
#     "../datasets/libero_goal/put_the_bowl_on_the_plate_demo",
#     "../datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo",
# ]

class MultiModalDataset(Dataset):
    def __init__(self, dataset_name_list):
        self.total_len = 0
        hdf5_file = dataset_name_list[0] + ".hdf5"
        f = h5py.File(hdf5_file, "r")
        demo_num = len(f['data'].keys())
        print("demo_num:", demo_num)

        self.agentview_images = safe_cuda(torch.from_numpy(f["data/demo_0/obs/agentview_rgb"][()].transpose(0, 3, 1, 2)))
        self.eye_in_hand_images = safe_cuda(torch.from_numpy(f["data/demo_0/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2)))
        joint_states = safe_cuda(torch.from_numpy(f["data/demo_0/obs/joint_states"][()]))
        gripper_states = safe_cuda(torch.from_numpy(f["data/demo_0/obs/gripper_states"][()]))
        ee_states = safe_cuda(torch.from_numpy(f["data/demo_0/obs/ee_states"][()]))
        self.proprio_states = torch.cat([joint_states, gripper_states, ee_states], dim=1)
        self.total_len += f["data/demo_0"].attrs["num_samples"]

        for i in range(1,demo_num):
            self.agentview_images = torch.cat([self.agentview_images, safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2)))], dim=0)
            self.eye_in_hand_images = torch.cat([self.eye_in_hand_images, safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2)))], dim=0)
            joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/joint_states"][()]))
            gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/gripper_states"][()]))
            ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/ee_states"][()]))
            self.proprio_states = torch.cat([self.proprio_states, torch.cat([joint_states, gripper_states, ee_states], dim=1)], dim=0)
            self.total_len += f[f"data/demo_{i}"].attrs["num_samples"]     
        f.close()
        print("Finish loading", hdf5_file)

        for j in range(1, len(dataset_name_list)):
            hdf5_file = dataset_name_list[j] + ".hdf5"
            f = h5py.File(hdf5_file, "r")
            demo_num = len(f['data'].keys())

            for k in range(demo_num):
                self.agentview_images = torch.cat([self.agentview_images, safe_cuda(torch.from_numpy(f[f"data/demo_{k}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2)))], dim=0)
                self.eye_in_hand_images = torch.cat([self.eye_in_hand_images, safe_cuda(torch.from_numpy(f[f"data/demo_{k}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2)))], dim=0)
                joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{k}/obs/joint_states"][()]))
                gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{k}/obs/gripper_states"][()]))
                ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{k}/obs/ee_states"][()]))
                self.proprio_states = torch.cat([self.proprio_states, torch.cat([joint_states, gripper_states, ee_states], dim=1)], dim=0)
                self.total_len += f[f"data/demo_{k}"].attrs["num_samples"]   
            f.close()
            print("Finish loading", hdf5_file)

        print("Total length of dataset:", self.total_len)



    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):

        proprio_state = self.proprio_states[idx].float()
        agentview_image = self.agentview_images[idx].float()
        eye_in_hand_image = self.eye_in_hand_images[idx].float()
        
        return proprio_state, agentview_image, eye_in_hand_image

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name',
        type=str,
        default="pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '--alpha-kl',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--alpha-force',
        type=float,
        default=1.0,
    )
    parser.add_argument(
            '--z-dim',
            type=int,
            default=32
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["agentview", "eye_in_hand", "proprio"]
    )

    parser.add_argument(
        "--no-skip",
        action="store_true"
    )

    parser.add_argument(
        '--use-checkpoint',
        action="store_true"
    )

    args = parser.parse_args()
    
    dataset = MultiModalDataset(Dataset_Name_List)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    data = []
    modalities = args.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{args.alpha_kl}"

    if args.no_skip:
        modality_str += "_no_skip"
    proprio_dim = 7 + 2 + 6 # joint_states + gripper_states + ee_states
    sensor_fusion = safe_cuda(SensorFusion(z_dim=args.z_dim, use_skip_connection=not args.no_skip, modalities=modalities, proprio_dim=proprio_dim))

    if args.use_checkpoint:
        sensor_fusion.load_state_dict(torch.load(f"results/{args.exp_name}/repr/Fusion_{modality_str}_{args.z_dim}_checkpoint.pth"))
        print("loaded checkpoint")
    optimizer = torch.optim.Adam(sensor_fusion.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=50)

    if args.no_skip:
        reduction = 'sum'
    else:
        reduction = 'mean'
    
    reconstruction_loss = torch.nn.MSELoss(reduction=reduction)
    bce_loss = torch.nn.BCELoss(reduction=reduction)
    n_epochs = args.num_epochs

    os.makedirs(f"results/{args.exp_name}/repr/imgs", exist_ok=True)
    last_loss = None
    for epoch in range(n_epochs):
        sensor_fusion.train()
        training_loss = 0
        for (proprio, img_1, img_2) in dataloader:

            proprio = safe_cuda(proprio)
            img_1 = safe_cuda(img_1 / 255.)
            img_2 = safe_cuda(img_2 / 255.)

            x = Modality_input(frontview=None, agentview=img_1, eye_in_hand=img_2, force=None, proprio=proprio)


            output, mu_z, var_z, mu_prior, var_prior = sensor_fusion(x)

            k = mu_z.size()[1]
            if args.no_skip:
                
                loss = args.alpha_kl * torch.sum(kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0)))
            else:
                loss = args.alpha_kl * torch.mean(kl_normal(mu_z, var_z, mu_prior.squeeze(0), var_prior.squeeze(0)))

            if 'agentview' in modalities:
                loss = loss + reconstruction_loss(img_1, output.agentview_recon)
            if 'eye_in_hand' in modalities:
                loss = loss + reconstruction_loss(img_2, output.eye_in_hand_recon)
            if 'force' in modalities:
                loss = loss + args.alpha_force * bce_loss(output.contact.squeeze(1), contact_state)
            if 'proprio' in modalities:
                loss = loss + reconstruction_loss(proprio, output.proprio) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        print(f"Training loss: {training_loss}")
            
        if epoch % 10 == 0:
            sensor_fusion.eval()
            proprio = safe_cuda(proprio)

            x = Modality_input(frontview=None, agentview=img_1, eye_in_hand=img_2, force=None, proprio=proprio)

            output, _, _, _, _ = sensor_fusion(x)
            
            if 'agentview' in modalities and 'eye_in_hand' in modalities:
                output_agentview = (output.agentview_recon * 255.).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                output_eye_in_hand = (output.eye_in_hand_recon * 255.).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

                example_agentview_img = []
                example_eye_in_hand_img = []

                for i in range(output_agentview.shape[0]):
                    example_agentview_img.append(output_agentview[i])
                    example_eye_in_hand_img.append(output_eye_in_hand[i])

                final_img_agentview = np.concatenate(example_agentview_img, axis=1)
                final_img_eye_in_hand = np.concatenate(example_eye_in_hand_img, axis=1)

                if epoch % 100 == 0:
                    # rgb -> bgr
                    final_img_agentview = cv2.cvtColor(final_img_agentview, cv2.COLOR_RGB2BGR)
                    final_img_eye_in_hand = cv2.cvtColor(final_img_eye_in_hand, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"results/{args.exp_name}/repr/imgs/test_agentview_img_{epoch}.png", final_img_agentview)
                    cv2.imwrite(f"results/{args.exp_name}/repr/imgs/test_eye_in_hand_img_{epoch}.png", final_img_eye_in_hand)

            if last_loss is None:
                last_loss = training_loss
            elif last_loss > training_loss:
                print("Saving checkpoint")
                torch.save(sensor_fusion.state_dict(), f"results/{args.exp_name}/repr/Fusion_{modality_str}_{args.z_dim}_checkpoint.pth")
                last_loss = training_loss
        scheduler.step(training_loss)
        if optimizer.param_groups[0]['lr'] < 1e-4:
            print("Learning rate became too low, stop training")
            break
    torch.save(sensor_fusion.state_dict(), f"results/{args.exp_name}/repr/Fusion_{modality_str}_{args.z_dim}.pth")
    print(f"Final saved loss: {training_loss}")

    # Save the embedding
    sensor_fusion = safe_cuda(SensorFusion(z_dim=args.z_dim, use_skip_connection=not args.no_skip, modalities=modalities, proprio_dim=proprio_dim))
    sensor_fusion.load_state_dict(torch.load(f"results/{args.exp_name}/repr/Fusion_{modality_str}_{args.z_dim}_checkpoint.pth"))
    print("loaded checkpoint")
    sensor_fusion.eval()

    for dataset_name in Dataset_Name_List:
        dataset_hdf5_file = dataset_name + ".hdf5"
        f = h5py.File(dataset_hdf5_file, "r")
        demo_num = len(f['data'].keys())

        dataset_name_parts = dataset_name.split("/")
        part_2 = dataset_name_parts[-2]
        part_1 = dataset_name_parts[-1]
        embedding_name = f"results/{args.exp_name}/repr/{part_2}/{part_1}/embedding_{modality_str}_{args.z_dim}.hdf5"
        os.makedirs(os.path.dirname(embedding_name), exist_ok=True)
        print("Saving embedding to", embedding_name)
        embedding_file = h5py.File(embedding_name, "w")
        grp = embedding_file.create_group("data")

        for i in range(demo_num):
            agentview_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/agentview_rgb"][()].transpose(0, 3, 1, 2))).float()
            eye_in_hand_images = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()].transpose(0, 3, 1, 2))).float()
            joint_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/joint_states"][()])).float()
            gripper_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/gripper_states"][()])).float()
            ee_states = safe_cuda(torch.from_numpy(f[f"data/demo_{i}/obs/ee_states"][()])).float()
            proprio_states = torch.cat([joint_states, gripper_states, ee_states], dim=1).float()

            proprio = safe_cuda(proprio_states)
            img_1 = safe_cuda(agentview_images / 255.)
            img_2 = safe_cuda(eye_in_hand_images / 255.)

            x = Modality_input(frontview=None, agentview=img_1, eye_in_hand=img_2, force=None, proprio=proprio)
            with torch.no_grad():
                embeddings = sensor_fusion(x, encoder_only=True).cpu().unsqueeze(1).numpy().astype('float32') # (Length, 1, z_dim)

            demo_data_grp = grp.create_group(f"demo_{i}")
            demo_data_grp.create_dataset("embedding", data=embeddings)

        embedding_file.close()
        f.close()





if __name__ == "__main__":
    main()