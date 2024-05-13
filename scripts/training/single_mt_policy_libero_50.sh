export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp25 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-pretrain25 lifelong.eval_in_train=False eval.eval=False

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp30 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er30 lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp25/Multitask/BCTransformerPolicy_seed1/single-pretrain25_run_001/multitask_model_ep50.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp35 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er35 lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp30/Multitask/BCTransformerPolicy_seed1/single-er30_run_001/multitask_model_ep50.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp40 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er40 lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp35/Multitask/BCTransformerPolicy_seed1/single-er35_run_001/multitask_model_ep50.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp45 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er45 lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp40/Multitask/BCTransformerPolicy_seed1/single-er40_run_001/multitask_model_ep50.pth

export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0 && python lifelong/main_old.py seed=1 benchmark_name=libero_kitchen_exp50 \
policy=bc_transformer_policy lifelong=multitask \
exp=single-er50 lifelong.eval_in_train=False eval.eval=False \
pretrain_model_path=experiments/libero_kitchen_exp45/Multitask/BCTransformerPolicy_seed1/single-er45_run_001/multitask_model_ep50.pth