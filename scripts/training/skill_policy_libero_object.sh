export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_object_exp6 \
skill_learning.exp_name=dinov2_libero_object_image_only_6 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-pretrain6 goal_modality=BUDS

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_object_exp7 \
skill_learning.exp_name=dinov2_libero_object_image_only_7 policy=bc_transformer_policy lifelong=multitask_skill goal_modality=BUDS \
exp=BUDS-single-er7 pretrain_model_path=experiments/libero_object_exp6/Multitask_Skill/dinov2_libero_object_image_only_6_seed1/BUDS-single-pretrain6_run_001

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_object_exp8 \
skill_learning.exp_name=dinov2_libero_object_image_only_8 policy=bc_transformer_policy lifelong=multitask_skill goal_modality=BUDS \
exp=BUDS-single-er8 pretrain_model_path=experiments/libero_object_exp7/Multitask_Skill/dinov2_libero_object_image_only_7_seed1/BUDS-single-er7_run_001

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_object_exp9 \
skill_learning.exp_name=dinov2_libero_object_image_only_9 policy=bc_transformer_policy lifelong=multitask_skill goal_modality=BUDS \
exp=BUDS-single-er9 pretrain_model_path=experiments/libero_object_exp8/Multitask_Skill/dinov2_libero_object_image_only_8_seed1/BUDS-single-er8_run_001

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_object_exp10 \
skill_learning.exp_name=dinov2_libero_object_image_only_10 policy=bc_transformer_policy lifelong=multitask_skill goal_modality=BUDS \
exp=BUDS-single-er10 pretrain_model_path=experiments/libero_object_exp9/Multitask_Skill/dinov2_libero_object_image_only_9_seed1/BUDS-single-er9_run_001