import pickle
import argparse
import h5py
import os
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.manifold import TSNE

import init_path
from models.args_utils import get_common_args, update_json_config

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from easydict import EasyDict
from hydra.experimental import compose, initialize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from collections import defaultdict, Counter
from PIL import Image

DEL_COST = INS_COST = 1.0
SUB_COST = 1.0
dataset_name_list = [
    "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo",
    "libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo",
]
base_dataset_name_list = dataset_name_list[0:6]
lifelong_dataset_name_list = dataset_name_list[6:10]



class Segment():
    def __init__(self, ep_idx, start_idx, end_idx, label=None):
        self.ep_idx = ep_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label

def cosine_distance(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return 1 - dot_product / (norm_A * norm_B)

def ComputeEditDistance(a: list, b: list):

    a_len = len(a) + 1
    b_len = len(b) + 1
    
    dist = np.zeros((a_len, b_len))

    for i in range(b_len):
        dist[0, i] = i
    for i in range(a_len):
        dist[i, 0] = i
    
    for i in range(1, a_len):
        for j in range(1, b_len):
            a_idx = i
            b_idx = j
            min_val = 10000
            cost = 0
            if b_idx > 0 and min_val > dist[a_idx][b_idx - 1]:
                min_val = dist[a_idx][b_idx-1]
                cost = DEL_COST
            if a_idx > 0 and b_idx > 0 and min_val > dist[a_idx-1][b_idx-1]:
                min_val = dist[a_idx-1][b_idx-1]
                cost = SUB_COST
                if a[i - 1] == b[j - 1]:
                    cost = 0
                
            if a_idx > 0 and min_val > dist[a_idx-1][b_idx]:
                min_val = dist[a_idx-1][b_idx]
                cost = INS_COST
            dist[a_idx][b_idx] = min_val + cost
    return dist[-1][-1]

def take_start_idx(elem):
    return elem[0]

def seg_start_idx(elem):
    return elem.start_idx

def segment_footprint(start_idx, end_idx, embeddings, mode="mean"):
    centroid_idx = (start_idx + end_idx) // 2
    if mode == "mean":
        return np.mean([embeddings[start_idx],
                        embeddings[centroid_idx],
                        embeddings[end_idx]], axis=0)

    elif mode == "head":
        return embeddings[start_idx]

    elif mode == "tail":
        return embeddings[end_idx]

    elif mode == "centroid":
        return embeddings[centroid_idx]

    elif mode == "concat_1":
        return np.concatenate([embeddings[start_idx],
                               embeddings[centroid_idx],
                               embeddings[end_idx]], axis=1)
    elif mode == "concat_2":
        return np.concatenate([embeddings[start_idx],
                               embeddings[(start_idx + centroid_idx) // 2],
                               embeddings[centroid_idx],
                               embeddings[(centroid_idx + end_idx) // 2],
                               embeddings[end_idx]], axis=1)
    
    elif mode == "global_pooling":
        return np.mean(embeddings[start_idx:end_idx+1], axis=0).reshape(1, -1)

    elif mode == "concat_8":
        # Calculate the step for getting 8 frames
        step = (end_idx - start_idx) // 7
        selected_frames = [embeddings[start_idx + i * step] for i in range(8)]
        return np.concatenate(selected_frames, axis=1)

    else:
        raise ValueError("Unknown mode!")
    
def assert_wrong_labeling(ep_subtasks_seq):
    for ep_idx in ep_subtasks_seq:
        ep_subtasks_seq[ep_idx].sort(key=seg_start_idx)

    for ep_idx in ep_subtasks_seq:
        for i in range(len(ep_subtasks_seq[ep_idx])-1):
            if ep_subtasks_seq[ep_idx][i].end_idx != ep_subtasks_seq[ep_idx][i+1].start_idx:
                print(f"Ep idx: {ep_idx}, seg idx: {i}")
                import pdb; pdb.set_trace()


def agglomoration_func(cfg, modality_str, dataset_name_list):
    X = []
    locs = []
    init_depth = 0
    dataset_idx = -1
    for dataset_name in dataset_name_list:
        dataset_idx += 1
        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        demo_num = len(h5py_file['data'].keys())
        embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
        embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
        with open(f"results/{cfg.exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
            trees = pickle.load(f)

        if cfg.repr.no_skip:
            scale = cfg.agglomoration.scale
        else:
            scale = 1.0
        for (ep_idx, tree) in trees["trees"].items():
            embeddings = embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()]

            depth = init_depth
            node_list = []
            while len(node_list) <  cfg.agglomoration.segment_scale: # cfg.agglomoration.K  * cfg.agglomoration.segment_scale: #TODO: #4
                node_list = tree.find_midlevel_abstraction(tree.root_node.idx, depth=depth, min_len=0)
                depth += 1

            for node_idx in node_list:
                node = tree.nodes[node_idx]
                embedding = segment_footprint(node.start_idx, node.end_idx, embeddings, cfg.agglomoration.segment_footprint) * scale
                X.append(embedding.squeeze())
                locs.append((dataset_idx, ep_idx, node_idx))

        h5py_file.close()
        embedding_h5py_f.close()



    K = cfg.agglomoration.K
    # range_n_clusters = list(range(8, 17)) 
    # silhouette_scores = []
    # for n_clusters in range_n_clusters:
    #     clustering_model = cluster.SpectralClustering(n_clusters=n_clusters,
    #                                                 assign_labels="discretize",
    #                                                 affinity=cfg.agglomoration.affinity)
    #     cluster_labels = clustering_model.fit_predict(X)
    #     silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    #     silhouette_scores.append(silhouette_avg)
    # optimal_k = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    # print(f"Optimal number of clusters (K) is: {optimal_k}")

    colors = ['r', 'b', 'g', 'y', 'k', 'C0', 'C1', 'C2', 'magenta', 'lightpink', 'deepskyblue', 'lawngreen'] + list(mcolors.CSS4_COLORS.values())

    if cfg.agglomoration.affinity != "kmeans":
        clustering = cluster.SpectralClustering(n_clusters=K,
                                                assign_labels="discretize",
                                                affinity=cfg.agglomoration.affinity# "nearest_neighbors"
                                                ).fit(X)

    else:
        clustering = cluster.KMeans(n_clusters=K,
                                    random_state=0).fit(X)

    labels = clustering.labels_
    print(clustering.get_params())

    
    loc_dict = {}
    for (loc, label) in zip(locs, labels):
        if loc[0] not in loc_dict:
            loc_dict[loc[0]] = {}
        if loc[1] not in loc_dict[loc[0]]:
            loc_dict[loc[0]][loc[1]] = []
        loc_dict[loc[0]][loc[1]].append((loc[2], label))

    merge_nodes = []
    # max_len = 0
    for dataset_idx in loc_dict:
        for ep_idx in loc_dict[dataset_idx]:
            previous_label = None
            start_idx = None
            end_idx = None
            for (loc, label) in loc_dict[dataset_idx][ep_idx]:
                dataset_name = dataset_name_list[dataset_idx]
                with open(f"results/{cfg.exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
                    trees = pickle.load(f)
                node = trees["trees"][ep_idx].nodes[loc]
                if previous_label is None:
                    previous_label = label
                    start_idx = node.start_idx
                    end_idx = node.end_idx
                else:
                    if previous_label == label and node.end_idx > start_idx:
                        end_idx = node.end_idx
                    else:
                        merge_nodes.append([dataset_idx, ep_idx, start_idx, end_idx, previous_label])
                        # if end_idx > max_len:
                        #     max_len = end_idx
                        previous_label = label
                        start_idx = node.start_idx
                        end_idx = node.end_idx
            merge_nodes.append([dataset_idx, ep_idx, start_idx, end_idx, label])

    # ----------------Process start -----------------
    # cluster again after merge        
    ep_subtasks_seq = {}

    max_len = 0
    X = []
    locs = []

    for dataset_idx, ep_idx, start_idx, end_idx, label in merge_nodes:
        if dataset_idx not in ep_subtasks_seq:
            ep_subtasks_seq[dataset_idx] = {}
        if ep_idx not in ep_subtasks_seq[dataset_idx]:
            ep_subtasks_seq[dataset_idx][ep_idx] = []
        ep_subtasks_seq[dataset_idx][ep_idx].append(Segment(ep_idx, start_idx, end_idx, label))
        if end_idx > max_len:
            max_len = end_idx

    for dataset_idx in ep_subtasks_seq:
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            ep_subtasks_seq[dataset_idx][ep_idx].sort(key=seg_start_idx)

    for dataset_idx in ep_subtasks_seq:
        dataset_name = dataset_name_list[dataset_idx]
        embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
        embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            for seg in ep_subtasks_seq[dataset_idx][ep_idx]:
                locs.append((dataset_idx, seg.ep_idx, seg.start_idx, seg.end_idx))
                embedding = segment_footprint(seg.start_idx, seg.end_idx, embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()], cfg.agglomoration.segment_footprint) * scale
                X.append(embedding.squeeze())
        embedding_h5py_f.close()

    if cfg.agglomoration.affinity != "kmeans":
        clustering = cluster.SpectralClustering(n_clusters=K,
                                                assign_labels="discretize",
                                                affinity=cfg.agglomoration.affinity).fit(X)
    else:
        clustering = cluster.KMeans(n_clusters=K,
                                    random_state=0).fit(X)
        print("Using K means")

    # If there is a cluster whose average length is shorter than a
    # threshold, merge the segments and decrease the number of
    # cluster, redo clustering again


    # Reorder label
    label_mapping = []
    labels = clustering.labels_    
    for (loc, label) in zip(locs, labels):
        if label not in label_mapping and len(label_mapping) < K:
            label_mapping.append(label)
    new_labels = []
    for label in labels:
        new_labels.append(label_mapping.index(label))
    labels = new_labels


    subtask_len_clusters = {}
    for label in range(K):
        subtask_len_clusters[label] = []

    cluster_indices = []
    for (loc, label) in zip(locs, labels):
        subtask_len_clusters[label].append(loc[3] - loc[2])
        cluster_indices.append(label)

    cluster_indices = np.array(cluster_indices)

    counter = 0
    for dataset_idx in ep_subtasks_seq:
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            for seg in ep_subtasks_seq[dataset_idx][ep_idx]:
                seg.label = labels[counter]
                counter += 1

    min_len_thresh = cfg.agglomoration.min_len_thresh
    remove_labels = []
    for (label, c) in subtask_len_clusters.items():
        print(label, np.mean(c))
        if np.mean(c) < min_len_thresh:
            remove_labels.append(label)
            print(colors[label])
    print(remove_labels)

    has_removed_labels = False

    while not has_removed_labels:
        new_ep_subtasks_seq = {}
        for dataset_idx in ep_subtasks_seq:
            new_ep_subtasks_seq[dataset_idx] = {}
            for ep_idx, ep_subtask_seq in ep_subtasks_seq[dataset_idx].items():
                new_ep_subtask_seq = []
                previous_label = None
                for idx in range(len(ep_subtask_seq)):
                    if previous_label is None:
                        previous_label = ep_subtask_seq[idx].label
                        start_idx = ep_subtask_seq[idx].start_idx
                        end_idx = ep_subtask_seq[idx].end_idx
                    else:
                        if previous_label == ep_subtask_seq[idx].label and ep_subtask_seq[idx].end_idx > start_idx:
                            end_idx = ep_subtask_seq[idx].end_idx
                        else:
                            new_ep_subtask_seq.append(Segment(ep_idx, start_idx, end_idx, previous_label))
                            previous_label = ep_subtask_seq[idx].label
                            start_idx = ep_subtask_seq[idx].start_idx
                            end_idx = ep_subtask_seq[idx].end_idx

                new_ep_subtask_seq.append(Segment(ep_idx, start_idx, end_idx, ep_subtask_seq[idx].label))
                new_ep_subtasks_seq[dataset_idx][ep_idx] = new_ep_subtask_seq
            
        ep_subtasks_seq = new_ep_subtasks_seq
        new_ep_subtasks_seq = {}
        for dataset_idx in ep_subtasks_seq:
            dataset_name = dataset_name_list[dataset_idx]
            embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
            embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
            with open(f"results/{cfg.exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
                trees = pickle.load(f)
            new_ep_subtasks_seq[dataset_idx] = {}
            for ep_idx, ep_subtask_seq in ep_subtasks_seq[dataset_idx].items():
                new_ep_subtask_seq = []
                idx = 0
                embeddings = embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()]
                while idx < len(ep_subtask_seq):
                    d1 = None
                    d2 = None
                    if idx == 0:
                        d2 = 0
                    elif idx == len(ep_subtask_seq) - 1:
                        d1 = 0
                    else:
                        d1 = trees["trees"][ep_idx].compute_distance(segment_footprint(ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx].end_idx, embeddings, cfg.agglomoration.segment_footprint),
                                                                    segment_footprint(ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx-1].end_idx, embeddings, cfg.agglomoration.segment_footprint))

                        d2 = trees["trees"][ep_idx].compute_distance(segment_footprint(ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx].end_idx, embeddings, cfg.agglomoration.segment_footprint),
                                                                    segment_footprint(ep_subtask_seq[idx+1].start_idx, ep_subtask_seq[idx+1].end_idx, embeddings, cfg.agglomoration.segment_footprint))
                    if ep_subtask_seq[idx].label in remove_labels: #  or (ep_subtask_seq[idx].end_idx - ep_subtask_seq[idx].start_idx) <= min_len_thresh:
                        if len(ep_subtask_seq) == 1:
                            print("remove data with only one remove label segment")
                        # Merge
                        elif d1 is None:
                            # Merge with after
                            new_seg = Segment(ep_idx, ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx+1].end_idx, ep_subtask_seq[idx+1].label)
                            new_ep_subtask_seq.append(new_seg)
                            step = 2
                        elif d2 is None:
                            # Merge with before
                            new_seg = Segment(ep_idx, ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx].end_idx, ep_subtask_seq[idx-1].label)
                            if new_ep_subtask_seq[-1].start_idx == new_seg.start_idx:
                                new_ep_subtask_seq.pop()
                            new_ep_subtask_seq.append(new_seg)
                            step = 1
                        else:
                            if d1 < d2:
                                new_seg = Segment(ep_idx, ep_subtask_seq[idx-1].start_idx, ep_subtask_seq[idx].end_idx, ep_subtask_seq[idx-1].label)
                                if new_ep_subtask_seq[-1].end_idx > new_seg.start_idx:
                                    new_seg.start_idx = new_ep_subtask_seq[-1].end_idx
                                if new_ep_subtask_seq[-1].start_idx == new_seg.start_idx:
                                    new_ep_subtask_seq.pop()
                                new_ep_subtask_seq.append(new_seg)
                                step = 1
                            else:
                                new_seg = Segment(ep_idx, ep_subtask_seq[idx].start_idx, ep_subtask_seq[idx+1].end_idx, ep_subtask_seq[idx+1].label)
                                new_ep_subtask_seq.append(new_seg)
                                step = 2
                    else:
                        new_ep_subtask_seq.append(ep_subtask_seq[idx])                    
                        step = 1
                    
                    idx += step

                new_ep_subtasks_seq[dataset_idx][ep_idx] = []
                previous_label = None
                start_idx = None
                end_idx = None
                
                for i in range(len(new_ep_subtask_seq)):
                    label = new_ep_subtask_seq[i].label
                    if previous_label is None:
                        start_idx = new_ep_subtask_seq[i].start_idx
                        end_idx = new_ep_subtask_seq[i].end_idx
                        previous_label = new_ep_subtask_seq[i].label
                    else:
                        if previous_label == label and new_ep_subtask_seq[i].end_idx > start_idx:
                            end_idx = new_ep_subtask_seq[i].end_idx
                        else:
                            new_ep_subtasks_seq[dataset_idx][ep_idx].append(Segment(ep_idx, start_idx, end_idx, previous_label))
                            previous_label = label
                            start_idx = new_ep_subtask_seq[i].start_idx
                            end_idx = new_ep_subtask_seq[i].end_idx
                if start_idx == None and end_idx == None:
                    del new_ep_subtasks_seq[dataset_idx][ep_idx]
                else:
                    new_ep_subtasks_seq[dataset_idx][ep_idx].append(Segment(ep_idx, start_idx, end_idx, label)) # note for start_idx and end_idx = 0, this is the wrong label

            embedding_h5py_f.close()
        ep_subtasks_seq = new_ep_subtasks_seq

        has_removed_labels = True
        for dataset_idx in ep_subtasks_seq:
            for ep_idx in ep_subtasks_seq[dataset_idx]:
                for idx in range(len(ep_subtasks_seq[dataset_idx][ep_idx])):
                    if ep_subtasks_seq[dataset_idx][ep_idx][idx].label in remove_labels:
                        has_removed_labels = False
                        break
                if not has_removed_labels:
                    break
            if not has_removed_labels:
                break
        
    K = K - len(remove_labels)
    for dataset_idx in ep_subtasks_seq: 
        print(ep_subtasks_seq[dataset_idx].keys())
    # ----------------Process finished -----------------
        plt.figure()
        for (ep_idx, ep_subtask_seq) in ep_subtasks_seq[dataset_idx].items():
            for seg in ep_subtask_seq:
                # if seg.start_idx == None and seg.end_idx == None:
                #     continue
                plt.plot([seg.start_idx, seg.end_idx], [seg.ep_idx, seg.ep_idx], colors[seg.label], linewidth=3)

        plt.xlabel("Sequence")
        plt.ylabel("No. Demo")

        dataset_name = dataset_name_list[dataset_idx]
        os.makedirs(f"results/{cfg.exp_name}/skill_data_vis/{dataset_name}", exist_ok=True)
        plt.savefig(f"results/{cfg.exp_name}/skill_data_vis/{dataset_name}/{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.png")
        plt.close()

        # plt.show()

        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        plt.figure()

        from PIL import Image
        mod_idx = 0
        while mod_idx < 4:
            mod_idx += 1
            fig = plt.figure(figsize=(100, 50))
            scale = 3.
            plt.xlim([0, (max_len) * scale])
            plt.ylim([-5, demo_num + 5])
            # plt.tight_layout()
            ax = plt.gca()
            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            img_size = 0.10

            image_info = []
            for (ep_idx, ep_subtask_seq) in ep_subtasks_seq[dataset_idx].items():
                if ep_idx % (demo_num // 5) != mod_idx:
                    continue
                for seg in ep_subtask_seq:
                    start_idx, end_idx, label = seg.start_idx, seg.end_idx, seg.label
                    point = plt.plot([start_idx * scale, end_idx * scale], [ep_idx, ep_idx], colors[label], linewidth=3)
                    xa, ya = trans2(trans([point[0].get_data()[0][0], point[0].get_data()[1][0]]))
                    image_info.append((xa, ya, start_idx, ep_idx))

                xa, ya = trans2(trans([point[0].get_data()[0][1], point[0].get_data()[1][1]]))
                image_info.append((xa, ya, end_idx, ep_idx))



            for info in image_info:
                xa, ya, start_idx, ep_idx = info
                # agentview_image = np.array(Image.open(h5py_file[f"data/demo_{ep_idx}/agentview_image_names"][()][start_idx]))
                agentview_image = np.array(h5py_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx])
                # inverse the img for better visualization
                agentview_image = np.flip(agentview_image, axis=0)
                new_axis = plt.axes([xa - img_size / 2, ya + img_size / 50, img_size, img_size])
                new_axis.imshow(agentview_image)
                new_axis.set_aspect('equal')
                new_axis.axis('off')



            plt.xlabel("Sequence")
            plt.ylabel("No. Demo")
            ax.axis('off')
            plt.savefig(f"results/{cfg.exp_name}/skill_data_vis/{dataset_name}/{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}_image.png")
            plt.close()

            # plt.show()
        
        h5py_file.close()

    if cfg.agglomoration.visualization:       
        exit()





    X = []
    final_colors = {}
    for dataset_idx in ep_subtasks_seq: 
        dataset_name = dataset_name_list[dataset_idx]
        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        demo_num = len(h5py_file['data'].keys())
        embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
        embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")

        dataset_category = dataset_name.split("/")[0]
        os.makedirs(f"results/{cfg.exp_name}/skill_data/{dataset_category}", exist_ok=True)
        subtask_file_name = f"results/{cfg.exp_name}/skill_data/{dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.hdf5"
        print(subtask_file_name)
        subtask_file = h5py.File(subtask_file_name, "w")

        grp = subtask_file.create_group("subtasks")
        grp.attrs["num_subtasks"] = K

        subtasks_grps = []
        subtasks = {}

        label_mapping = []
        for i in range(cfg.agglomoration.K):
            if i not in remove_labels:
                label_mapping.append(i)
        for i in range(K):
            subtasks_grps.append(grp.create_group(f"subtask_{i}"))
            subtasks[i] = []
            final_colors[i] = colors[label_mapping[i]]

        for ep_idx, ep_subtask_seq in ep_subtasks_seq[dataset_idx].items():
            for seg in ep_subtask_seq:
                subtasks[label_mapping.index(seg.label)].append([ep_idx, seg.start_idx, seg.end_idx])

                # save final (embedding,label)
                embedding = segment_footprint(seg.start_idx, seg.end_idx, embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()], cfg.agglomoration.segment_footprint) * cfg.agglomoration.scale
                X.append((embedding.squeeze(),label_mapping.index(seg.label), dataset_idx, ep_idx, seg.start_idx, seg.end_idx))

        # print(ep_subtasks_seq[1])
        for i in range(K):
            subtasks_grps[i].create_dataset("segmentation", data=subtasks[i])

        ep_strings = []
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            subtask_seq = ep_subtasks_seq[dataset_idx][ep_idx]
            subtask_seq.sort(key=seg_start_idx)

            ep_string = []
            for subtask in subtask_seq:
                ep_string.append(subtask.label)
            ep_strings.append(ep_string)
            # print(f"{ep_idx}: {ep_string}")
        assert_wrong_labeling(ep_subtasks_seq[dataset_idx])
        
        saved_ep_subtasks_seq = {}
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            ep_subtasks_seq[dataset_idx][ep_idx].sort(key=seg_start_idx)
            saved_ep_subtasks_seq[dataset_idx] = []
            prev_seg = None
            for seg in ep_subtasks_seq[dataset_idx][ep_idx]:
                saved_ep_subtasks_seq[dataset_idx].append([seg.start_idx, seg.end_idx, label_mapping.index(seg.label)])
                seg.label = label_mapping.index(seg.label)
                if prev_seg is not None:
                    assert(seg.start_idx == prev_seg.end_idx)
                prev_seg = seg
            grp.create_dataset(f"demo_subtasks_seq_{ep_idx}", data=saved_ep_subtasks_seq[dataset_idx])

        grp.attrs["demo_num"] = demo_num
        
        for ep_idx in ep_subtasks_seq[dataset_idx]:
            for i in range(len(ep_subtasks_seq[dataset_idx][ep_idx])-1):
                if ep_subtasks_seq[dataset_idx][ep_idx][i].end_idx != ep_subtasks_seq[dataset_idx][ep_idx][i+1].start_idx:
                    print(f"Ep idx: {ep_idx}, seg idx: {i}")
                    import pdb; pdb.set_trace()
        
            
        print(f"Final K: {K}")

        score = 0.

        num_pairs = 0
        for i in range(len(ep_strings)):
            for j in range(len(ep_strings)):
                if i >= j:
                    continue
                dist = ComputeEditDistance(ep_strings[i], ep_strings[j])
                score += dist
                num_pairs += 1


        score /= ((len(ep_strings) * (len(ep_strings) - 1)) / 2)
        print(score)
        grp.attrs["score"] = score
        
        subtask_file.close()
        h5py_file.close()
        embedding_h5py_f.close()



    print(final_colors)

    embeddings = [x[0] for x in X]
    cluster_labels = [x[1] for x in X]
    task_ids = [x[2] for x in X]
    demo_indices = [x[3] for x in X]
    seg_start = [x[4] for x in X]
    seg_end = [x[5] for x in X]
    with h5py.File(f"results/{cfg.exp_name}/skill_data/saved_feature_data.hdf5", 'w') as hf:
        hf.create_dataset('embeddings', data=np.stack(embeddings))
        hf.create_dataset('cluster_labels', data=cluster_labels)
        hf.create_dataset('task_ids', data=task_ids)
        hf.create_dataset('demo_indices', data=demo_indices)
        hf.create_dataset('seg_start', data=seg_start)
        hf.create_dataset('seg_end', data=seg_end)
    X = [(e, l, d, ep) for (e, l, d, ep, _, _) in X]
    for lifelong_dataset_name in lifelong_dataset_name_list:
        print(f"add new data: {lifelong_dataset_name}")
        ep_subtasks_seq, X, dataset_name_list = add_new_data(cfg, modality_str, ep_subtasks_seq, X, dataset_name_list, lifelong_dataset_name)
        embeddings = [x[0] for x in X]
        cluster_labels = [x[1] for x in X]
        task_ids = [x[2] for x in X]
        demo_indices = [x[3] for x in X]
        seg_start = [x[4] for x in X]
        seg_end = [x[5] for x in X]
        save_exp_name = f"{cfg.exp_name}_{len(dataset_name_list)}"
        with h5py.File(f"results/{save_exp_name}/skill_data/saved_feature_data.hdf5", 'w') as hf:
            hf.create_dataset('embeddings', data=np.stack(embeddings))
            hf.create_dataset('cluster_labels', data=cluster_labels)
            hf.create_dataset('task_ids', data=task_ids)
            hf.create_dataset('demo_indices', data=demo_indices)
            hf.create_dataset('seg_start', data=seg_start)
            hf.create_dataset('seg_end', data=seg_end)
        X = [(e, l, d, ep) for (e, l, d, ep, _, _) in X]

def find_medoid(cluster_data, distance_fn):
    if distance_fn == "l2":
        distances = np.sum(np.linalg.norm(cluster_data[:, None] - cluster_data, axis=2), axis=1)
    elif distance_fn == "cosine":
        distances = np.sum(1 - (np.dot(cluster_data, cluster_data.T) / 
                               (np.linalg.norm(cluster_data, axis=1)[:, None] * np.linalg.norm(cluster_data, axis=1))), axis=1)
    else:
        raise ValueError("Unknown distance function")
    return cluster_data[np.argmin(distances)]

def compute_small_centers(X, distance_fn):
    cluster_dataset_centers = defaultdict(lambda: defaultdict(list))
    
    # Group features by cluster label and dataset_idx
    for feature, label, dataset_idx, _ in X:
        cluster_dataset_centers[label][dataset_idx].append(feature)
    
    # Calculate the center for each cluster and dataset_idx as the point with the minimum sum of distances
    for label, dataset_dict in cluster_dataset_centers.items():
        for dataset_idx, features in dataset_dict.items():
            min_distance_sum = float('inf')
            best_center = None
            for feature in features:
                if distance_fn == "l2":
                    distance_sum = np.sum([np.linalg.norm(feature - other_feature) for other_feature in features])
                elif distance_fn == "cosine":
                    distance_sum = np.sum([1 - np.dot(feature, other_feature) / (np.linalg.norm(feature) * np.linalg.norm(other_feature)) for other_feature in features])
                else:
                    raise ValueError(f"Unsupported distance function: {distance_fn}")
                
                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    best_center = feature
            dataset_dict[dataset_idx] = best_center
    
    return cluster_dataset_centers

def assign_or_create_new_cluster(data, ep_idx, X, distance_fn, dataset_idx):
    # Compute small centers
    small_centers = compute_small_centers(X, distance_fn)
    
    # Compute the size of each small center
    center_sizes = defaultdict(lambda: defaultdict(int))
    for _, label, ds_idx, _ in X:
        center_sizes[label][ds_idx] += 1
    
    # Sort all small centers based on their distance to the new data
    distances_and_centers = []
    for label, dataset_dict in small_centers.items():
        for ds_idx, center in dataset_dict.items():
            if distance_fn == "l2":
                distance = np.linalg.norm(data - center)
            elif distance_fn == "cosine":
                distance = 1 - np.dot(data, center) / (np.linalg.norm(data) * np.linalg.norm(center))
            else:
                raise ValueError(f"Unsupported distance function: {distance_fn}")
            
            distances_and_centers.append((distance, label, ds_idx))
    
    distances_and_centers.sort(key=lambda x: x[0])  # sort by distance
    
    # Find the nearest valid small center for the new data
    nearest_label = None
    nearest_dataset_idx = None
    nearest_distance = None
    for distance, label, ds_idx in distances_and_centers:
        # Check if there are more than 1 small center for the cluster and the size of the current center is less than 10
        if center_sizes[label][ds_idx] >= 5 or len(small_centers[label]) == 1:
            # if len(small_centers[label]) == 1:
            #     import ipdb; ipdb.set_trace()
            #     print("?")
            nearest_distance = distance
            nearest_label = label
            nearest_dataset_idx = ds_idx
            break
    
    # Compute maximum distance between small centers of the nearest label
    centers_of_nearest_label = list(small_centers[nearest_label].values())
    if len(centers_of_nearest_label) > 1:
        if distance_fn == "l2":
            distances_between_centers = euclidean_distances(centers_of_nearest_label)
        elif distance_fn == "cosine":
            distances_between_centers = cosine_distances(centers_of_nearest_label)
        max_distance_between_centers = np.max(distances_between_centers)
    else:
        max_distance_between_centers = 1
    
    # If the nearest distance is greater than max_distance_between_centers, create a new cluster
    if nearest_distance > max_distance_between_centers:
        print(f"Create new cluster")
        new_label = max([label for _, label, _, _ in X]) + 1
        X.append((data, new_label, dataset_idx, ep_idx))
        return X, new_label
    else:
        X.append((data, nearest_label, dataset_idx, ep_idx))
        return X, nearest_label
    
def reassign_small_clusters(X, old_K, distance_fn, threshold=10):
    small_centers = compute_small_centers(X, distance_fn)
    
    # Compute the size of each cluster
    cluster_sizes = defaultdict(int)
    for _, label, _, _ in X:
        cluster_sizes[label] += 1
    
    # Compute the size of each small center
    center_sizes = defaultdict(lambda: defaultdict(int))
    for _, label, ds_idx, _ in X:
        center_sizes[label][ds_idx] += 1
    
    updated_X = []
    threshold = 15 #40 # 15 for libero object and goal 100 for libero10
    # For each data point in X
    for data, label, ds_idx, ep_idx in X:
        # If the data point is from a new cluster and its size is below the threshold
        if label >= old_K and cluster_sizes[label] < threshold:
            # Sort all small centers based on their distance to the data
            distances_and_centers = []
            for center_label, dataset_dict in small_centers.items():
                for center_ds_idx, center in dataset_dict.items():
                    # Exclude the data's own small center
                    if label == center_label and ds_idx == center_ds_idx:
                        continue
                    # avoid to assign to other small clusters
                    if center_label >= old_K and center_sizes[center_label][center_ds_idx] < threshold:
                        continue
                    if distance_fn == "l2":
                        distance = np.linalg.norm(data - center)
                    elif distance_fn == "cosine":
                        distance = 1 - np.dot(data, center) / (np.linalg.norm(data) * np.linalg.norm(center))
                    else:
                        raise ValueError(f"Unsupported distance function: {distance_fn}")
                    
                    distances_and_centers.append((distance, center_label, center_ds_idx))
            
            distances_and_centers.sort(key=lambda x: x[0])  # sort by distance
            
            # Find the nearest valid small center for the data
            nearest_label = None
            for distance, center_label, center_ds_idx in distances_and_centers:
                if center_sizes[center_label][center_ds_idx] >= 10 or len(small_centers[center_label]) == 1:
                    nearest_label = center_label
                    break
            
            # Reassign the data point to the nearest valid cluster
            updated_X.append((data, nearest_label, ds_idx, ep_idx))
        else:
            # If the data point is from an old cluster or from a new cluster with size above the threshold, keep it unchanged
            updated_X.append((data, label, ds_idx, ep_idx))
    
    return updated_X

def make_labels_continuous(X):
    # Extract unique labels and sort them
    unique_labels = sorted(set(label for _, label, _, _ in X))
    
    # Create a mapping from old label to new continuous label
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Update the labels in X using the mapping
    updated_X = [(data, label_mapping[label], ds_idx, ep_idx) for data, label, ds_idx, ep_idx in X]
    
    return updated_X

def visualize_tsne(tsne_results, labels, title, colors_list, name, cfg, dataset_name, new_dataset_idx, xlim=None, ylim=None):
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label < len(colors_list):  # Ensure the label has a corresponding color
            indices = np.where(labels == label)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors_list[label], alpha=0.8, label=f'Cluster {label}')

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels)  # Show custom legend
    plt.title(title)
    plt.grid(True)
    # Set the xlim and ylim if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # plt.show()
    save_exp_name = f"{cfg.exp_name}_{new_dataset_idx+1}"
    os.makedirs(f"results/{save_exp_name}/skill_data_vis/{dataset_name}", exist_ok=True)
    plt.savefig(f"results/{save_exp_name}/skill_data_vis/{dataset_name}/debug_{name}.png")
    # plt.close()

def add_new_data(cfg, modality_str, old_ep_subtasks_seq, X, old_dataset_name_list, dataset_name):
    print("-"*20 + "Add new data" + "-"*20)
    dataset_name_list = old_dataset_name_list + [dataset_name]
    new_dataset_idx = len(old_dataset_name_list)
    X_original = X
    old_K = max(label for _, label, _, _ in X_original) + 1
    colors = ['r', 'b', 'g', 'y', 'k', 'C0', 'C1', 'C2', 'magenta', 'lightpink', 'deepskyblue', 'lawngreen'] + list(mcolors.CSS4_COLORS.values())

    feature = []
    locs = []
    init_depth = 0
    dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
    h5py_file = h5py.File(dataset_name_path, "r")
    demo_num = len(h5py_file['data'].keys())
    embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
    embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
    with open(f"results/{cfg.exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
        trees = pickle.load(f)

    if cfg.repr.no_skip:
        scale = cfg.agglomoration.scale
    else:
        scale = 1.0
    for (ep_idx, tree) in trees["trees"].items():
        embeddings = embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()]

        depth = init_depth
        node_list = []
        while len(node_list) < cfg.agglomoration.segment_scale: #cfg.agglomoration.K  * cfg.agglomoration.segment_scale: #8
            node_list = tree.find_midlevel_abstraction(tree.root_node.idx, depth=depth, min_len=0)
            depth += 1

        for node_idx in node_list:
            node = tree.nodes[node_idx]
            embedding = segment_footprint(node.start_idx, node.end_idx, embeddings, cfg.agglomoration.segment_footprint) * scale
            feature.append((embedding.squeeze(), ep_idx))
            locs.append((ep_idx, node_idx))

    h5py_file.close()
    embedding_h5py_f.close()

    # visualize_tsne(X, title="T-SNE Visualization Before Processing", colors=colors, name="before", cfg=cfg, dataset_name=dataset_name)
    labels_l2 = []
    labels_cosine = []
    X = X_original.copy()
    print("length of feature: ", len(feature))
    labels_all = [label for _, label, _, _ in X]
    label_counts = Counter(labels_all)
    print(label_counts)
    for data in feature:
        X, label_cosine = assign_or_create_new_cluster(data[0], data[1], X, "cosine", new_dataset_idx)
        labels_cosine.append(label_cosine)

    labels_all = [label for _, label, _, _ in X]
    label_counts = Counter(labels_all)
    print(label_counts)

    X = reassign_small_clusters(X, old_K, "cosine")
    X = make_labels_continuous(X)
    K = max(label for _, label, _, _ in X) + 1
    print('K: ', K)
    labels_all = [label for _, label, _, _ in X]
    label_counts = Counter(labels_all)
    print(label_counts)

    ## merge nodes
    loc_dict = {}
    for (loc, label) in zip(locs, labels_all[len(X_original):]):
        if loc[0] not in loc_dict:
            loc_dict[loc[0]] = []
        loc_dict[loc[0]].append((loc[1], label))

    merge_nodes = []
    for ep_idx in loc_dict:
        previous_label = None
        start_idx = None
        end_idx = None
        for (loc, label) in loc_dict[ep_idx]:
            with open(f"results/{cfg.exp_name}/skill_classification/trees/{dataset_name}_trees_{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}.pkl", "rb") as f:
                trees = pickle.load(f)
            node = trees["trees"][ep_idx].nodes[loc]
            if previous_label is None:
                previous_label = label
                start_idx = node.start_idx
                end_idx = node.end_idx
            else:
                if previous_label == label and node.end_idx > start_idx:
                    end_idx = node.end_idx
                else:
                    merge_nodes.append([ep_idx, start_idx, end_idx, previous_label])
                    previous_label = label
                    start_idx = node.start_idx
                    end_idx = node.end_idx
        merge_nodes.append([ep_idx, start_idx, end_idx, label])

    ep_subtasks_seq = {}
    max_len = 0
    new_feature = []
    new_label = []
    locs = []

    for ep_idx, start_idx, end_idx, label in merge_nodes:
        if ep_idx not in ep_subtasks_seq:
            ep_subtasks_seq[ep_idx] = []
        ep_subtasks_seq[ep_idx].append(Segment(ep_idx, start_idx, end_idx, label))
        if end_idx > max_len:
            max_len = end_idx

    for ep_idx in ep_subtasks_seq:
        ep_subtasks_seq[ep_idx].sort(key=seg_start_idx)

    embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
    embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")
    for ep_idx in ep_subtasks_seq:
        for seg in ep_subtasks_seq[ep_idx]:
            if seg.label >= K:
                continue
            locs.append((seg.ep_idx, seg.start_idx, seg.end_idx))
            embedding = segment_footprint(seg.start_idx, seg.end_idx, embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()], cfg.agglomoration.segment_footprint) * scale
            new_feature.append((embedding.squeeze(), ep_idx))
            new_label.append(seg.label)
    embedding_h5py_f.close()

    all_ep_subtasks_seq = old_ep_subtasks_seq.copy()
    all_ep_subtasks_seq[new_dataset_idx] = ep_subtasks_seq

    X_processed = X_original.copy()
    for feature, label in zip(new_feature, new_label):
        X_processed.append((feature[0], label, new_dataset_idx, feature[1]))

    # Extract features and labels
    X_features = np.array([feature for feature, label, _, _ in X_processed])
    X_labels = [label for feature, label, _, _ in X_processed]



    # visualization

    # Calculate t-SNE for the combined data
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_features)

    # Visualization using both original and new data
    visualize_tsne(tsne_results, X_labels, "T-SNE Visualization of Processed Data", colors_list=colors, name="processed", cfg=cfg, dataset_name=dataset_name, new_dataset_idx=new_dataset_idx)

    # Retrieve the xlim and ylim for the processed data
    xlim = plt.xlim()
    ylim = plt.ylim()
    # Visualization using original data only
    visualize_tsne(tsne_results[:len(X_original)], X_labels[:len(X_original)], "T-SNE Visualization of Original Data", colors_list=colors, name="original", cfg=cfg, dataset_name=dataset_name, new_dataset_idx=new_dataset_idx, xlim=xlim, ylim=ylim)




    # K = max(X_labels) + 1
    for dataset_idx in all_ep_subtasks_seq: 
        # print(all_ep_subtasks_seq[dataset_idx].keys())
    # ----------------Process finished -----------------
        plt.figure()
        for (ep_idx, ep_subtask_seq) in all_ep_subtasks_seq[dataset_idx].items():
            for seg in ep_subtask_seq:
                # if seg.start_idx == None and seg.end_idx == None:
                #     continue
                plt.plot([seg.start_idx, seg.end_idx], [seg.ep_idx, seg.ep_idx], colors[seg.label], linewidth=3)

        plt.xlabel("Sequence")
        plt.ylabel("No. Demo")

        dataset_name = dataset_name_list[dataset_idx]
        save_exp_name = f"{cfg.exp_name}_{len(dataset_name_list)}"
        os.makedirs(f"results/{save_exp_name}/skill_data_vis/{dataset_name}", exist_ok=True)
        plt.savefig(f"results/{save_exp_name}/skill_data_vis/{dataset_name}/{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.png")
        plt.close()

        # plt.show()

        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        plt.figure()

        from PIL import Image
        mod_idx = 0
        while mod_idx < 4:
            mod_idx += 1
            fig = plt.figure(figsize=(100, 50))
            scale = 3.
            plt.xlim([0, (max_len) * scale])
            plt.ylim([-5, demo_num + 5])
            # plt.tight_layout()
            ax = plt.gca()
            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            img_size = 0.10

            image_info = []
            for (ep_idx, ep_subtask_seq) in all_ep_subtasks_seq[dataset_idx].items():
                if ep_idx % (demo_num // 5) != mod_idx:
                    continue
                for seg in ep_subtask_seq:
                    start_idx, end_idx, label = seg.start_idx, seg.end_idx, seg.label
                    point = plt.plot([start_idx * scale, end_idx * scale], [ep_idx, ep_idx], colors[label], linewidth=3)
                    xa, ya = trans2(trans([point[0].get_data()[0][0], point[0].get_data()[1][0]]))
                    image_info.append((xa, ya, start_idx, ep_idx))

                xa, ya = trans2(trans([point[0].get_data()[0][1], point[0].get_data()[1][1]]))
                image_info.append((xa, ya, end_idx, ep_idx))



            for info in image_info:
                xa, ya, start_idx, ep_idx = info
                # agentview_image = np.array(Image.open(h5py_file[f"data/demo_{ep_idx}/agentview_image_names"][()][start_idx]))
                agentview_image = np.array(h5py_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx])
                # inverse the img for better visualization
                agentview_image = np.flip(agentview_image, axis=0)
                new_axis = plt.axes([xa - img_size / 2, ya + img_size / 50, img_size, img_size])
                new_axis.imshow(agentview_image)
                new_axis.set_aspect('equal')
                new_axis.axis('off')



            plt.xlabel("Sequence")
            plt.ylabel("No. Demo")
            ax.axis('off')
            plt.savefig(f"results/{save_exp_name}/skill_data_vis/{dataset_name}/{modality_str}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}_image.png")
            plt.close()

            # plt.show()
        
        h5py_file.close()




    ## save data
    X = []
    for dataset_idx in all_ep_subtasks_seq: 
        dataset_name = dataset_name_list[dataset_idx]
        dataset_name_path = "../datasets/" + dataset_name + ".hdf5"
        h5py_file = h5py.File(dataset_name_path, "r")
        demo_num = len(h5py_file['data'].keys())
        embedding_hdf5_path = f"results/{cfg.exp_name}/repr/{dataset_name}/embedding_{modality_str}_{cfg.repr.z_dim}.hdf5"
        embedding_h5py_f = h5py.File(embedding_hdf5_path, "r")

        dataset_category = dataset_name.split("/")[0]
        save_exp_name = f"{cfg.exp_name}_{len(dataset_name_list)}"
        os.makedirs(f"results/{save_exp_name}/skill_data/{dataset_category}", exist_ok=True)
        subtask_file_name = f"results/{save_exp_name}/skill_data/{dataset_name}_subtasks_{modality_str}_{cfg.repr.z_dim}_{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{K}_{cfg.agglomoration.affinity}.hdf5"
        print(subtask_file_name)
        subtask_file = h5py.File(subtask_file_name, "w")

        grp = subtask_file.create_group("subtasks")
        grp.attrs["num_subtasks"] = K

        subtasks_grps = []
        subtasks = {}

        for i in range(K):
            subtasks_grps.append(grp.create_group(f"subtask_{i}"))
            subtasks[i] = []

        for ep_idx, ep_subtask_seq in all_ep_subtasks_seq[dataset_idx].items():
            for seg in ep_subtask_seq:
                if seg.label in subtasks:
                    subtasks[seg.label].append([ep_idx, seg.start_idx, seg.end_idx])

                    # save final (embedding,label)
                    embedding = segment_footprint(seg.start_idx, seg.end_idx, embedding_h5py_f[f"data/demo_{ep_idx}/embedding"][()], cfg.agglomoration.segment_footprint) * cfg.agglomoration.scale
                    X.append((embedding.squeeze(), seg.label, dataset_idx, ep_idx, seg.start_idx, seg.end_idx))

        # print(all_ep_subtasks_seq[1])
        for i in range(K):
            subtasks_grps[i].create_dataset("segmentation", data=subtasks[i])

        ep_strings = []
        for ep_idx in all_ep_subtasks_seq[dataset_idx]:
            subtask_seq = all_ep_subtasks_seq[dataset_idx][ep_idx]
            subtask_seq.sort(key=seg_start_idx)

            ep_string = []
            for subtask in subtask_seq:
                ep_string.append(subtask.label)
            ep_strings.append(ep_string)
            # print(f"{ep_idx}: {ep_string}")
        assert_wrong_labeling(all_ep_subtasks_seq[dataset_idx])
        
        saved_ep_subtasks_seq = {}
        for ep_idx in all_ep_subtasks_seq[dataset_idx]:
            all_ep_subtasks_seq[dataset_idx][ep_idx].sort(key=seg_start_idx)
            saved_ep_subtasks_seq[dataset_idx] = []
            prev_seg = None
            for seg in all_ep_subtasks_seq[dataset_idx][ep_idx]:
                saved_ep_subtasks_seq[dataset_idx].append([seg.start_idx, seg.end_idx, seg.label])
                if prev_seg is not None:
                    assert(seg.start_idx == prev_seg.end_idx)
                prev_seg = seg
            grp.create_dataset(f"demo_subtasks_seq_{ep_idx}", data=saved_ep_subtasks_seq[dataset_idx])

        grp.attrs["demo_num"] = demo_num
        
        for ep_idx in all_ep_subtasks_seq[dataset_idx]:
            for i in range(len(all_ep_subtasks_seq[dataset_idx][ep_idx])-1):
                if all_ep_subtasks_seq[dataset_idx][ep_idx][i].end_idx != all_ep_subtasks_seq[dataset_idx][ep_idx][i+1].start_idx:
                    print(f"Ep idx: {ep_idx}, seg idx: {i}")
                    import pdb; pdb.set_trace()
        
            
        print(f"Final K: {K}")
        score = 0.
        num_pairs = 0
        for i in range(len(ep_strings)):
            for j in range(len(ep_strings)):
                if i >= j:
                    continue
                dist = ComputeEditDistance(ep_strings[i], ep_strings[j])
                score += dist
                num_pairs += 1

        score /= ((len(ep_strings) * (len(ep_strings) - 1)) / 2)
        print(score)
        grp.attrs["score"] = score
        
        subtask_file.close()
        h5py_file.close()
        embedding_h5py_f.close()
    
    return all_ep_subtasks_seq, X, dataset_name_list


@hydra.main(config_path="../../configs/skill_learning", config_name="default", version_base=None)
def main(hydra_cfg):
    
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    
    print(f"Footprint: {cfg.agglomoration.footprint}, Dist: {cfg.agglomoration.dist}, Segment: {cfg.agglomoration.segment_footprint}, K: {cfg.agglomoration.K}, Affinity: {cfg.agglomoration.affinity}")

    modality_str =cfg.modality_str

    agglomoration_func(cfg, modality_str, base_dataset_name_list)

if __name__ == "__main__":
    main()
