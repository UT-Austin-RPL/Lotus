"""Hierarchical agglomoration"""

import os
import h5py
import shutil
import pickle
import init_path
import yaml
import hydra
from skill_discovery.hierarchical_agglomoration_utils import Node, HierarchicalAgglomorativeTree, save_agglomorative_tree
from omegaconf import OmegaConf
from easydict import EasyDict


def filter_labels(labels):
    for i in range(len(labels)):
        # In the beginning
        if i < 3:
            if labels[i+1] == labels[i+2] == labels[i+3] and labels[i] != labels[i+1]:
                labels[i] = labels[i+1]
        # At tail
        elif len(labels)-3 < i < len(labels) - 1:
            if labels[i-1] == labels[i-2] == labels[i-3] and labels[i] != labels[i-1]:
                labels[i] = labels[i-1]
        elif 3 <= i <= len(labels) - 3:
            # label = find_most_frequent_element(labels)
            if (labels[i-1] == labels[i-2] == labels[i+1] or labels[i-1] == labels[i+1] == labels[i+2]) and (labels[i-1] != labels[i]):
                labels[i] = labels[i-1]
    return labels


@hydra.main(config_path="../../configs/skill_learning", config_name="default", version_base=None)
def main(hydra_cfg):
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    exp_name = cfg.exp_name
    
    print(f"Footprint mode: {cfg.agglomoration.footprint}, Dist mode: {cfg.agglomoration.dist}")

    modality_str = cfg.modality_str
    # cfg.repr.z_dim = 768*2

    exp_dir = f"results/{cfg.exp_name}/repr"
    dataset_name_list = []
    for dataset_category in os.listdir(exp_dir):
        dataset_category_path = os.path.join(exp_dir, dataset_category)
        if os.path.isdir(dataset_category_path) and dataset_category in ['libero_object','libero_spactial','libero_goal', 'libero_10', 'libero_90', 'rw_all']:
            for dataset_name in os.listdir(dataset_category_path):
                dataset_name_path = os.path.join(dataset_category_path, dataset_name)
                if os.path.isdir(dataset_name_path):
                    dataset_name_list.append(f"{dataset_category}/{dataset_name}")
    
    for dataset_name in dataset_name_list:
        dataset_category = dataset_name.split("/")[0]
        dir_name = f"results/{cfg.exp_name}/skill_classification/trees/{dataset_category}"
        os.makedirs(dir_name, exist_ok=True)
        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        demo_num = len(h5py_file['data'].keys())
        print(f"dataset_name:{dataset_name}, demo_num:{demo_num}")

        embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
        embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
    
        total_len = 0

        try:
            shutil.rmtree(f"results/{cfg.exp_name}/skill_classification/initial_clustering")
        except:
            pass
            
        step = cfg.agglomoration.agglomoration_step
        trees = {"trees": {}}

        for demo_idx in range(demo_num):
            embeddings = embedding_h5py_f[f"data/demo_{demo_idx}/embedding"][()]
            agentview_images = h5py_file[f"data/demo_{demo_idx}/obs/agentview_rgb"][()]
            # agentview_image_names_list = h5py_file[f"data/demo_{demo_idx}/agentview_image_names"][()]

            agglomorative_tree = HierarchicalAgglomorativeTree()

            agglomorative_tree.agglomoration(embeddings, step,
                                            footprint_mode=cfg.agglomoration.footprint,
                                            dist_mode=cfg.agglomoration.dist)
            agglomorative_tree.create_root_node()

            trees["trees"][demo_idx] = agglomorative_tree

            # Visualization
            save_agglomorative_tree(agglomorative_tree, agentview_images, demo_idx, exp_name, dataset_name,
                                    footprint_mode=cfg.agglomoration.footprint,
                                    dist_mode=cfg.agglomoration.dist,
                                    modality_mode=modality_str)
            
        trees["info"] = {"dataset_name": dataset_name, "demo_num": demo_num}

        with open(f"results/{exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "wb") as f:
            pickle.dump(trees, f)

        h5py_file.close()
        embedding_h5py_f.close()

if __name__ == "__main__":
    main()
