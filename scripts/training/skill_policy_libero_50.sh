export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp25 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_25 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-pretrain25 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp30 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_30 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-er30 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask_Skill/dinov2_libero_50_image_only_gp_K13_2_25_seed1/pretrain25_run_001

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp35 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_35 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-er35 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask_Skill/dinov2_libero_50_image_only_gp_K13_2_25_seed1/BUDS-single-er30

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp40 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_40 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-er40 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask_Skill/dinov2_libero_50_image_only_gp_K13_2_25_seed1/BUDS-single-er35

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp45 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_45 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-er45 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask_Skill/dinov2_libero_50_image_only_gp_K13_2_25_seed1/BUDS-single-er40

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main.py seed=1 benchmark_name=libero_kitchen_exp50 \
skill_learning.exp_name=dinov2_libero_50_image_only_gp_K13_2_50 policy=bc_transformer_policy lifelong=multitask_skill \
exp=BUDS-single-er50 goal_modality=BUDS lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask_Skill/dinov2_libero_50_image_only_gp_K13_2_25_seed1/BUDS-single-er45