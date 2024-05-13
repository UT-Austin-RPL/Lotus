import torch
from torchvision import transforms
import numpy as np
import cv2
import argparse
import h5py
import os
from functools import partial
from r3m import load_r3m

from einops import rearrange
from easydict import EasyDict
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from PIL import Image

from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.model_utils import safe_cuda

# import init_path
# from utils.video_utils import KaedeVideoWriter

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

# Dataset_Name_List = [
#     "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
#     "../datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
#     "../datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
#     "../datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
#     "../datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
#     "../datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
#     "../datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
# ]

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
#     "../datasets/libero_object/libero_object_40/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_ketchup_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_butter_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_milk_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo",
#     "../datasets/libero_object/libero_object_40/pick_up_the_orange_juice_and_place_it_in_the_basket_demo",
# ]

Dataset_Name_List = [
    "../datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo",
    "../datasets/libero_goal/put_the_bowl_on_the_stove_demo",
    "../datasets/libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo",
    "../datasets/libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo",
    "../datasets/libero_goal/put_the_bowl_on_top_of_the_cabinet_demo",
    "../datasets/libero_goal/push_the_plate_to_the_front_of_the_stove_demo",
    "../datasets/libero_goal/put_the_cream_cheese_in_the_bowl_demo",
    "../datasets/libero_goal/turn_on_the_stove_demo",
    "../datasets/libero_goal/put_the_bowl_on_the_plate_demo",
    "../datasets/libero_goal/put_the_wine_bottle_on_the_rack_demo",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name',
        type=str,
        default="debug",
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=2048*2,
    )
    parser.add_argument(
        '--modality-str',
        type=str,
        default="r3m_agentview_eye_in_hand",
    )
    args = parser.parse_args()
    modality_str = args.modality_str
    feature_dim = args.feature_dim
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from r3m import load_r3m
    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224),T.ToTensor()])

    for dataset_name in Dataset_Name_List:
        dataset_hdf5_file = dataset_name + ".hdf5"
        f = h5py.File(dataset_hdf5_file, "r")
        demo_num = len(f['data'].keys())

        dataset_name_parts = dataset_name.split("/")
        part_2 = dataset_name_parts[-2]
        part_1 = dataset_name_parts[-1]
        embedding_name = f"results/{args.exp_name}/repr/{part_2}/{part_1}/embedding_{modality_str}_{feature_dim}.hdf5"
        os.makedirs(os.path.dirname(embedding_name), exist_ok=True)
        print("Saving embedding to", embedding_name)
        embedding_file = h5py.File(embedding_name, "w")
        grp = embedding_file.create_group("data")

        for i in range(demo_num):
            agentview_images = f[f"data/demo_{i}/obs/agentview_rgb"][()]
            eye_in_hand_images = f[f"data/demo_{i}/obs/eye_in_hand_rgb"][()]

            # img_1 = safe_cuda(agentview_images / 255.)
            # img_2 = safe_cuda(eye_in_hand_images / 255.)

            transformed_images = [transforms(Image.fromarray(img.astype(np.uint8))) for img in agentview_images]
            agentview_images = torch.stack(transformed_images)

            agentview_images.to(device) 
            with torch.no_grad():
                agentview_features = r3m(agentview_images * 255.0) ## R3M expects image input to be [0-255]


            transformed_images = [transforms(Image.fromarray(img.astype(np.uint8))) for img in eye_in_hand_images]
            eye_in_hand_images = torch.stack(transformed_images)

            eye_in_hand_images.to(device) 
            with torch.no_grad():
                eye_in_hand_features = r3m(eye_in_hand_images * 255.0) ## R3M expects image input to be [0-255] 

            embeddings = torch.cat([agentview_features, eye_in_hand_features], dim=1).detach().cpu().unsqueeze(1).numpy().astype('float32')
            # 2048 * 2
            if np.isnan(embeddings).any():
                print("NAN")
                import ipdb; ipdb.set_trace()
            # embeddings = torch.cat([agentview_features, eye_in_hand_features, proprio], dim=1).cpu().unsqueeze(1).numpy().astype('float32')

            demo_data_grp = grp.create_group(f"demo_{i}")
            demo_data_grp.create_dataset("embedding", data=embeddings)

        embedding_file.close()
        f.close()