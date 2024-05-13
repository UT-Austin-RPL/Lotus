<div align="center">

<b>**LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery**</b>

[[Website]](https://ut-austin-rpl.github.io/Lotus/)
[[Paper]](https://arxiv.org/pdf/2311.02058)
______________________________________________________________________
</div>

**LOTUS** is a continual imitation learning algorithm that empowers a physical robot to continuously and efficiently learn to solve new manipulation tasks throughout its lifespan. The core idea behind **LOTUS** is constructing an ever-growing skill library from a sequence of new tasks with a small number of corresponding task demonstrations. **LOTUS** starts with a continual skill discovery process using an open-vocabulary vision model, which extracts skills as recurring patterns presented in unstructured demonstrations. Continual skill discovery updates existing skills to avoid catastrophic forgetting of previous tasks and adds new skills to exhibit novel behaviors. **LOTUS** trains a meta-controller that flexibly composes various skills to tackle vision-based manipulation tasks in the lifelong learning process.

---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Unsupervised Skill Discovery](#Unsupervised-Skill-Discovery)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)


# Installtion
First clone this repo, and then run the following commands in the given order to install the dependency for **LOTUS**.
```
conda create -n lotus python=3.9.19
conda activate lotus
cd Lotus
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

# Datasets
We use high-quality human teleoperation demonstrations for the four task suites from [**LIBERO**] (https://github.com/Lifelong-Robot-Learning/LIBERO). To download the demonstration dataset, run:
```python
python libero_benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python libero_benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.


# Getting Started

In the following, we provide example scripts for unsupervised skill discovery, training and evaluation.

## Unsupervised Skill Discovery

The following is a example of unsupervised skill discovery on `libero_object` dataset.

```shell
cd lotus/skill_learning
```

### Encoding Representation
```shell
python multisensory_repr/dinov2_repr.py  --exp-name dinov2_libero_object_image_only --modality-str dinov2_agentview_eye_in_hand --feature-dim 1536
```
Output: 
- `results/{exp_name}/repr/{DatasetCategoty}/{DatasetName}/embedding_{modality_str}_{feature_dim}.hdf5`

### Hierarchical Agglomerative Clustering
```shell
python skill_discovery/hierarchical_agglomoration.py exp_name=dinov2_libero_object_image_only modality_str=dinov2_agentview_eye_in_hand repr.z_dim=1536 agglomoration.dist=cos agglomoration.footprint=global_pooling
```
Output: 
- `results/{exp_name}/skill_classification/agglomoration_results/{DatasetCategoty}/{DatasetName}/{agglomoration.footprint}_{agglomoration.dist}_{modality_str}/{idx}.png`
- `results/{exp_name}/skill_classification/trees/{DatasetCategoty}/{dataset_name}_trees_{modality_str}_{agglomoration.footprint}_{agglomoration.dist}.pkl`

### Spectral Clustering
```shell
python skill_discovery/agglomoration_script.py exp_name=dinov2_libero_object_image_only modality_str=dinov2_agentview_eye_in_hand repr.z_dim=1536 agglomoration.segment_scale=1 agglomoration.min_len_thresh=30 agglomoration.K=2 agglomoration.scale=0.01 agglomoration.dist=cos
```
Output:
- `results/{exp_name}_{current_task_num}/skill_data/{DatasetCategoty}/{DatasetCategoty}/{DatasetName}_subtasks_{modality_str}_{feature_dim}_mean_{agglomoration.dist}_concat_1_K{agglomoration.K}_{cfg.agglomoration.affinity}.hdf5`



### Save Dinov2 Feature for Hierarchical Policy Training
```shell
cd lotus/skill_learning
python multisensory_repr/save_dinov2_repr.py
```
Output: 
- `../datasets/dinov2/{DatasetCategoty}/{DatasetName}.hdf5`

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[libero_object_exp6, ... ,LIBERO_KITCHEN_EXP50]` (Please see `lotus/libero/benchmark/__init__.py` for full registered benchmark name list)
- `EXP_NAME`: experiment name
- `SKILL_EXP_NAME`: experiment name in `Unsupervised Skill Discovery` (e.g., `dinov2_libero_object_image_only`)
- `PRETRAIN_MODEL_PATH`: pretrain model path

For single multitask policy training from scratch, run the following:
```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lotus/lifelong/main_old.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=bc_transformer_policy \
                               lifelong=multitask \
                               exp={EXP_NAME}
```
For single multitask policy finetuning, run the following:
```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lotus/lifelong/main_old.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=bc_transformer_policy \
                               lifelong=multitask \
                               exp={EXP_NAME} \
                               pretrain_model_path={PRETRAIN_MODEL_PATH}
```

For hierarchical skill-based policy training from scratch, run the following:
```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lotus/lifelong/main_old.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=bc_transformer_policy \
                               lifelong=multitask_skill \
                               skill_learning.exp_name={SKILL_EXP_NAME} \
                               exp={EXP_NAME} \
                               goal_modality=BUDS
```
For hierarchical skill-based policy finetuning, run the following:
```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lotus/lifelong/main_old.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=bc_transformer_policy \
                               lifelong=multitask_skill \
                               exp={EXP_NAME} \
                               pretrain_model_path={PRETRAIN_MODEL_PATH}
```
Currently, we haven't developed automated scripts for our method in the process of lifelong learning. At present, commands need to be run individually for different stages. Please see the scripts in `scripts/training` for full training command.


## Evaluation

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately. Please see `lotus/lifelong/evaluate_old.py` (for single multitask policy) and `lotus/lifelong/evaluate.py` (for hierarchical skill-based policy) for more details.

# Acknowledgement
The code base used in this project is sourced from these repository:

[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

[UT-Austin-RPL/BUDS](https://github.com/UT-Austin-RPL/BUDS)


# Citation
If you find **LOTUS** to be useful in your own research, please consider citing our paper:

```bibtex
@article{wan2024lotus,
  title={Lotus: Continual imitation learning for robot manipulation through unsupervised skill discovery},
  author={Wan, Weikang and Zhu, Yifeng and Shah, Rutav and Zhu, Yuke},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```
