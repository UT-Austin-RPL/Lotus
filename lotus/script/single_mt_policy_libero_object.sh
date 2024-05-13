export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp6 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-pretrain6

## ER

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp7 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er7 pretrain_model_path=experiments/libero_object_exp6/Multitask/BCTransformerPolicy_seed1/single-pretrain6_run_001/multitask_model.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp8 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er8 pretrain_model_path=experiments/libero_object_exp7/Multitask/BCTransformerPolicy_seed1/single-er7_run_001/multitask_model.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp9 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er9 pretrain_model_path=experiments/libero_object_exp8/Multitask/BCTransformerPolicy_seed1/single-er8_run_001/multitask_model.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp10 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er10 pretrain_model_path=experiments/libero_object_exp9/Multitask/BCTransformerPolicy_seed1/single-er9_run_001/multitask_model.pth


## FT

# export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp7 \
# policy=bc_transformer_policy lifelong=multitask \
# exp=single-ft7 pretrain_model_path=experiments/libero_object_exp6/Multitask/BCTransformerPolicy_seed1/single-pretrain6_run_001/multitask_model.pth

# export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp8 \
# policy=bc_transformer_policy lifelong=multitask \
# exp=single-ft8 pretrain_model_path=experiments/libero_object_exp7/Multitask/BCTransformerPolicy_seed1/single-ft7_run_001/multitask_model.pth

# export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp9 \
# policy=bc_transformer_policy lifelong=multitask \
# exp=single-ft9 pretrain_model_path=experiments/libero_object_exp8/Multitask/BCTransformerPolicy_seed1/single-ft8_run_001/multitask_model.pth

# export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_object_exp10 \
# policy=bc_transformer_policy lifelong=multitask \
# exp=single-ft10 pretrain_model_path=experiments/libero_object_exp9/Multitask/BCTransformerPolicy_seed1/single-ft9_run_001/multitask_model.pth