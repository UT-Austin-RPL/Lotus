import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import numpy as np
import cv2
from PIL import Image

class Node():
    def __init__(self, start_idx, end_idx, level, idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.level = level
        self.idx = idx

        self.parent_idx = None
        self.children_node_indices = []
        self.cluster_label = None

    @property
    def centroid_idx(self):
        return (self.start_idx + self.end_idx) // 2

    @property
    def len(self):
        return self.end_idx - self.start_idx
    
class HierarchicalAgglomorativeTree():
    """Construct a hierarchical tree for clusters

    """
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_indices = []
        self.level_indices = {}
        self.graph = nx.Graph()
        self.root_node = None

    def add_nodes(self, nodes):
        for node in nodes:
            if node.idx in self.node_indices:
                continue
            self.add_node(node)

    def add_node(self, node):
        self.nodes.append(node)
        if node.children_node_indices != []:
            for child_idx in node.children_node_indices:
                self.edges.append([node.idx, child_idx])
        self.node_indices.append(node.idx)
        if node.level not in self.level_indices.keys():
            self.level_indices[node.level] = []

        self.level_indices[node.level].append(node.idx)

    def to_nx_graph(self):
        for node in self.nodes:
            self.graph.add_node(node.idx)
            
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1])

        return self.graph

    def find_parent(self, node_idx):
        while self.nodes[node_idx].parent_idx is not None:
            node_idx = self.nodes[node_idx].parent_idx
        return self.nodes[node_idx]

    def create_root_node(self):
        root_node = Node(start_idx=0, end_idx=0, level=-1, idx=len(self.nodes))
        for node in self.nodes:
            if node.parent_idx is None:

                root_node.children_node_indices.append(node.idx)
                root_node.level = max(node.level + 1, root_node.level)
                node.parent_idx = root_node.idx
                self.edges.append([root_node.idx, node.idx])                

        self.nodes.append(root_node)
        self.root_node = root_node
        self.level_indices[self.root_node.level] = [self.root_node.idx]

    @property
    def max_depth(self):
        return max(self.level_indices.keys()) + 1

    
    def find_children_nodes(self, parent_node_idx, depth=0, no_leaf=False, min_len=20):
        # Return when depth = 0
        node_list = []
        for node_idx in self.nodes[parent_node_idx].children_node_indices:
            if depth == 0 or self.nodes[node_idx].len < min_len:
                node_list.append(node_idx)
            else:

                if no_leaf:
                    if self.nodes[node_idx].level == 1:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1)
                else:
                    if self.nodes[node_idx].level == 0 or self.nodes[node_idx].len < min_len:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1)

        return node_list

    def find_midlevel_abstraction(self, parent_node_idx, depth=0, no_leaf=False, min_len=40):
        # Return when depth = 0
        node_list = []
        for node_idx in self.nodes[parent_node_idx].children_node_indices:
            if depth == 0:
                node_list.append(node_idx)
            else:
                if no_leaf:
                    if self.nodes[node_idx].level == 1:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1, min_len=min_len)
                else:
                    if self.nodes[node_idx].level == 0:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1, min_len=min_len)

        return node_list

    def check_consistency(self, node_idx):
        node = self.nodes[node_idx]
        if node.level == 0:
            return True
        else:
            children_nodes = self.find_children_nodes(node_idx, 0)
            for child_idx in children_nodes:
                if node.cluster_label != self.nodes[child_idx].cluster_label:
                    return False
            return True
        
    
    def assign_labels(self, node_idx, label):
        self.nodes[node_idx].cluster_label = label

    def unassign_labels(self, node_idx):
        self.nodes[node_idx].cluster_label = None
        for child_idx in self.find_children_nodes(node_idx, 0):
            self.nodes[child_idx].cluster_label = None

    def compute_distance(self, e1, e2, mode="l2"):
        if mode == "l2":
            return np.linalg.norm(e1 - e2)
        elif mode == "l1":
            return np.linalg.norm((e1 - e2), ord=1)
        elif mode == "cos":
            cos_similarity = np.dot(e1, e2.transpose()) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            return 1 - cos_similarity
        elif mode == "js":
            mu_e1, var_e1 = np.split(e1, 2, axis=-1)
            mu_e2, var_e2 = np.split(e2, 2, axis=-1)
            def kl_normal(qm, qv, pm, pv):
                element_wise = 0.5 * (np.log(pv) - np.log(qv) + qv / pv + np.power(qm - pm, 2) / pv - 1)
                return element_wise.sum(-1)
            js_dist = 0.5 * (kl_normal(mu_e1, var_e1, mu_e2, var_e2) + kl_normal(mu_e2, var_e2, mu_e1, var_e1))
            return  js_dist

    def node_footprint(self, node: Node, embeddings, mode="mean"):
        if mode == "centroid":
            return embeddings[node.centroid_idx]
        elif mode == "mean":
            embedding = np.mean([embeddings[node.start_idx], embeddings[node.centroid_idx], embeddings[node.end_idx]], axis=0)
            return embedding
        elif mode == "head":
            return embeddings[node.start_idx]
        elif mode == "tail":
            return embeddings[node.end_idx]
        elif mode == "concat_1":
            return np.concatenate([embeddings[node.start_idx], embeddings[node.centroid_idx], embeddings[node.end_idx]], axis=1)
        elif mode == "gaussian":
            mu = np.mean(embeddings[node.start_idx:node.end_idx+1], axis=0)
            var = np.mean(np.square(embeddings[node.start_idx:node.end_idx+1]), axis=0) - mu ** 2 + 1e-5
            assert(np.all(mu.shape == var.shape))
            return np.concatenate([mu, var], axis=1)
        elif mode == "global_pooling":
            return np.mean(embeddings[node.start_idx:node.end_idx+1], axis=0).reshape(1, -1)

    def find_nn(self, embeddings, node_idx, before_idx, after_idx, footprint_mode="mean", dist_mode="l2"):
        f1 = self.node_footprint(self.nodes[node_idx], embeddings, mode=footprint_mode)
        f2 = self.node_footprint(self.nodes[before_idx], embeddings, mode=footprint_mode) 
        f3 = self.node_footprint(self.nodes[after_idx], embeddings, mode=footprint_mode)

        d1 = self.compute_distance(f1, f2, mode=dist_mode)
        d2 = self.compute_distance(f1, f3, mode=dist_mode)

        return d1 < d2

    def agglomoration(self, embeddings, step, footprint_mode="mean", dist_mode="l2", len_penalty=True):
        idx = 0
        nodes = []
        terminate = False
        for i in range(len(embeddings)-1):
            if i % step == 0:
                if (i + 2 * step >= len(embeddings)-1):
                    start_idx = i
                    end_idx = len(embeddings) - 1
                    terminate = True
                else:
                    start_idx = i
                    end_idx = min(i + step, len(embeddings) - 1)
                node = Node(start_idx=start_idx,
                            end_idx=end_idx,
                            level=0,
                            idx=idx)

                idx += 1
                nodes.append(node)
                if terminate:
                    break
        self.add_nodes(nodes)

        while len(nodes) > 2:
            i = 1
            dist_seq = []            
            for i in range(len(nodes) - 1):
                dist = self.compute_distance(self.node_footprint(nodes[i], embeddings, mode=footprint_mode),
                                                      self.node_footprint(nodes[i+1], embeddings, mode=footprint_mode), mode=dist_mode)
                if len_penalty:
                    # Very simple penalty
                    # dist += (nodes[i].len + nodes[i+1].len) * (1./
                    # 10.)
                    
                    # Pentaly with respect to the whole length
                    dist += (nodes[i].len + nodes[i+1].len) / (5. * len(nodes))
                dist_seq.append(dist)

            target_idx = dist_seq.index(min(dist_seq))

            new_node = Node(start_idx=nodes[target_idx].start_idx,
                            end_idx=nodes[target_idx+1].end_idx,
                            level=max(nodes[target_idx].level, nodes[target_idx+1].level) + 1,
                            idx=idx)
            new_node.children_node_indices = [nodes[target_idx].idx, nodes[target_idx + 1].idx]
            nodes[target_idx].parent_idx = idx
            nodes[target_idx + 1].parent_idx  = idx
            self.add_node(new_node)
            idx += 1

            new_nodes = []
            visited_nodes = []

            for node in nodes:
                parent_node = self.find_parent(node.idx)
                if parent_node.idx not in visited_nodes:
                    new_nodes.append(parent_node)
                    visited_nodes.append(parent_node.idx)
            nodes = new_nodes

def save_agglomorative_tree(agglomorative_tree, agentview_images, ep_idx, exp_name, dataset_name, footprint_mode, dist_mode, modality_mode):
    
        fig = plt.figure(figsize=(25, 10))   
        width = 100
        depth = 3
        x = y = 0

        positions = {}
        for (i, node_idx) in enumerate(agglomorative_tree.level_indices[0]):
            positions[node_idx] = [x + i * width, y]
            plt.plot([x + i * width, x + i * width],
                     [y, y + depth / 2],
                     'k')

        for level in range(1, agglomorative_tree.max_depth):
            y += depth
            for (i, node_idx) in enumerate(agglomorative_tree.level_indices[level]):
                child_node_x_positions = []
                weights = []

                min_x = 10000
                max_x = 0
                for child_idx in agglomorative_tree.nodes[node_idx].children_node_indices:
                    child_node_x_positions.append(positions[child_idx][0])
                    if min_x > positions[child_idx][0]:
                        min_x = positions[child_idx][0]
                    if max_x < positions[child_idx][0]:
                        max_x = positions[child_idx][0]

                    plt.plot([positions[child_idx][0], positions[child_idx][0]],
                             [positions[child_idx][1], y - depth / 2], 'k')

                plt.plot([min_x, max_x],
                         [y - depth / 2, y - depth/2], 'k')
                positions[node_idx] = [np.mean(child_node_x_positions), y]
                plt.plot([positions[node_idx][0], positions[node_idx][0]],
                         [y - depth / 2, y], 'k')
                
            num_nodes = len(agglomorative_tree.level_indices[level])

        points = {}

        min_x = min_y = 10000
        max_x = max_y = 0
        for node in agglomorative_tree.nodes:
            point = plt.plot(positions[node.idx][0], positions[node.idx][1], 'ko')
            points[node.idx] = (point[0].get_data()[0][0], point[0].get_data()[1][0])
            min_x = min(min_x, points[node.idx][0])
            max_x = max(max_x, points[node.idx][0])
            min_y = min(min_y, points[node.idx][1])
            max_y = max(max_y, points[node.idx][1])

        plt.xlim([min_x - 50, max_x + 50])
        plt.ylim([min_y - 5, max_y + 5])
        plt.axis('off')
        ax=plt.gca()
        fig=plt.gcf()

        label_pos = 0.5 # middle of edge, halfway between nodes
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        img_size = 0.08

        counter = 0
        for agglomorative_tree_node in agglomorative_tree.nodes:
            if agglomorative_tree_node.children_node_indices == []:
                if counter % 1 == 0: # Only plot every 3th image
                    xa, ya = trans2(trans(points[agglomorative_tree_node.idx]))

                    new_axis = plt.axes([xa - img_size / 2.0, ya - img_size * 1.1, img_size, img_size])

                    # img = np.array(Image.open(agentview_image_names_list[agglomorative_tree_node.centroid_idx]))
                    img = agentview_images[agglomorative_tree_node.centroid_idx]
                    # inverse the img for better visualization
                    img = np.flip(img, axis=0)
                    new_axis.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # new_axis.imshow(img)
                    new_axis.set_aspect('equal')
                    new_axis.axis('off')
                counter += 1

        canvas = FigureCanvas(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()        
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))

        os.makedirs(f"results/{exp_name}/skill_classification/agglomoration_results/{dataset_name}/{footprint_mode}_{dist_mode}_{modality_mode}", exist_ok=True)
        cv2.imwrite(f"results/{exp_name}/skill_classification/agglomoration_results/{dataset_name}/{footprint_mode}_{dist_mode}_{modality_mode}/{ep_idx}.png", image)
