from models.conf_utils import *
from easydict import EasyDict

def output_parent_dir_template(cfg):
    folder_path = cfg.folder
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/"    
    return output_dir

def single_subskill_path_template(cfg, subtask_id, use_changepoint=False):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)

    folder_path = cfg.folder
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/task_{cfg.multitask.training_task_id}_bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"
    if use_changepoint:
        output_dir = f"{output_dir}_changepoint"
        
    model_name =  f"{output_dir}/{goal_str}{data_modality_str}_{cfg.skill_subgoal_cfg.subgoal_type}_subtask_{subtask_id}.pth"        

    summary_writer_name = f"{output_dir}/{goal_str}{data_modality_str}_subtask_{subtask_id}"
    return EasyDict({"output_dir": output_dir,
                     "model_checkpoint_name": model_name,
                     "summary_writer_name": summary_writer_name})


def subskill_path_template(cfg, subtask_id, use_changepoint=False):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)

    folder_path = cfg.folder
    if cfg.skill_subgoal_cfg is None and cfg.skill_training.policy_type == "no_subgoal":
        output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.skill_training.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.agglomoration.policy_type}"
    else:
        output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"
        
    if use_changepoint:
        output_dir = f"{output_dir}_changepoint"
        
    model_name =  f"{output_dir}/{goal_str}{data_modality_str}_{cfg.skill_subgoal_cfg.subgoal_type}_subtask_{subtask_id}.pth"        

    summary_writer_name = f"{output_dir}/{goal_str}{data_modality_str}_subtask_{subtask_id}"
    return EasyDict({"output_dir": output_dir,
                     "model_checkpoint_name": model_name,
                     "summary_writer_name": summary_writer_name})

def single_skill_path_template(cfg):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)

    folder_path = cfg.folder
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"
    model_name =  f"{output_dir}/{goal_str}{data_modality_str}_single_skill.pth"        

    summary_writer_name = f"{output_dir}/{goal_str}{data_modality_str}_single"
    return EasyDict({"output_dir": output_dir,
                     "model_checkpoint_name": model_name,
                     "summary_writer_name": summary_writer_name})

def gti_path_template(cfg):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)

    folder_path = cfg.folder
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"
    model_name =  f"{output_dir}/{goal_str}{data_modality_str}_gti.pth"
    summary_writer_name = f"{output_dir}/{goal_str}{data_modality_str}_gti"
    return EasyDict({"output_dir": output_dir,
                     "model_checkpoint_name": model_name,
                     "summary_writer_name": summary_writer_name})

def rpl_path_template(cfg):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)

    folder_path = cfg.folder
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/bc_mlp_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"
    model_name =  f"{output_dir}/{goal_str}{data_modality_str}_rpl.pth"
    summary_writer_name = f"{output_dir}/{goal_str}{data_modality_str}_rpl"
    return EasyDict({"output_dir": output_dir,
                     "model_checkpoint_name": model_name,
                     "summary_writer_name": summary_writer_name})

    
def vae_path_template(cfg):
    data_modality_str = get_data_modality_str(cfg)
    modality_str = get_modalities_str(cfg)
    goal_str = get_goal_str(cfg)
    output_dir = f"results/{cfg.data.dataset_name}"
    model_name =  f"{output_dir}/subgoal_vae.pth"
    return output_dir, model_name

def meta_path_template(cfg):
    folder_path = cfg.folder
    modality_str = get_modalities_str(cfg)
    
    if cfg.skill_training.policy_type == "no_subgoal":
        output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.policy_type}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}"
    else:
        output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"

    if cfg.meta.use_spatial_softmax:
        spatial_softmax_str = "_spatial_softmax"
    else:
        spatial_softmax_str = ""

    if cfg.meta_cvae_cfg.enable:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}_cvae"
    else:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}"

    if cfg.meta.random_affine:
        model_name += "_data_aug"

    summary_writer_name = model_name
    model_name += ".pth"
        
    # summary_writer_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}"
    return EasyDict({"output_dir": output_dir,
                     "model_name": model_name,
                     "summary_writer_name": summary_writer_name})

def cp_meta_path_template(cfg):
    folder_path = cfg.folder
    modality_str = get_modalities_str(cfg)
    
    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/cp_meta_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}"

    if cfg.meta.use_spatial_softmax:
        spatial_softmax_str = "_spatial_softmax"
    else:
        spatial_softmax_str = ""

    if cfg.meta_cvae_cfg.enable:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}_cvae"
    else:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}"

    if cfg.meta.random_affine:
        model_name += "_data_aug"

    summary_writer_name = model_name
    model_name += ".pth"
        
    return EasyDict({"output_dir": output_dir,
                     "model_name": model_name,
                     "summary_writer_name": summary_writer_name})



def subgoal_embedding_path_template(cfg, modality_str):
    folder_path = cfg.folder
    subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_embedding.hdf5"
        
    return subgoal_embedding_file_name

def singletask_subgoal_embedding_path_template(cfg, modality_str):
    folder_path = cfg.folder
    if cfg.multitask.training_task_id == -1:
        if cfg.skill_cvae_cfg.enable:
            subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_cvae_{cfg.skill_subgoal_cfg.subgoal_type}_embedding.hdf5"
        else:
            subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_embedding.hdf5"
    else:
        print("Single task training")
        subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/task_{cfg.multitask.training_task_id}_{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_embedding.hdf5"
        
    return subgoal_embedding_file_name

def cp_singletask_subgoal_embedding_path_template(cfg, modality_str):
    folder_path = cfg.folder
    if cfg.multitask.training_task_id == -1:
        if cfg.skill_cvae_cfg.enable:
            subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_cvae_{cfg.skill_subgoal_cfg.subgoal_type}_embedding.hdf5"
        else:
            subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_embedding.hdf5"
    else:
        print("Single task training")
        subgoal_embedding_file_name = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/task_{cfg.multitask.training_task_id}_{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.agglomoration.K}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}_embedding.hdf5"
    
    return subgoal_embedding_file_name



def multitask_meta_path_template(cfg):
    folder_path = cfg.folder
    modality_str = get_modalities_str(cfg)

    if cfg.multitask.training_task_id == -1:
        # Train on individual initial configurations
        if cfg.skill_training.policy_type == "no_subgoal":
            output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}_{cfg.skill_training.policy_type}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}/{cfg.multitask.task_id}"
        else:
            output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}/{cfg.multitask.task_id}"
    else:
        output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}/task_{cfg.multitask.training_task_id}"


    if cfg.meta.use_spatial_softmax:
        spatial_softmax_str = "_spatial_softmax"
    else:
        spatial_softmax_str = ""

    if cfg.meta_cvae_cfg.enable:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}_cvae"
    else:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}"

    if cfg.multitask.testing_percentage < 1.0:
        model_name += f"_{cfg.multitask.testing_percentage}"
        
    if cfg.meta.random_affine:
        model_name += "_data_aug"

    summary_writer_name = model_name
    model_name += ".pth"
        
    return EasyDict({"output_dir": output_dir,
                     "model_name": model_name,
                     "summary_writer_name": summary_writer_name})



def singletask_multitask_meta_path_template(cfg):
    folder_path = cfg.folder
    modality_str = get_modalities_str(cfg)

    output_dir = folder_path + f"results/{cfg.data.dataset_name}/run_{cfg.skill_training.run_idx}/meta_policy_{modality_str}_{cfg.repr.z_dim}/{cfg.agglomoration.footprint}_{cfg.agglomoration.dist}_{cfg.agglomoration.segment_footprint}_K{cfg.skill_training.agglomoration.K}_{cfg.agglomoration.affinity}_{cfg.skill_training.policy_type}_horizon_{cfg.skill_subgoal_cfg.horizon}_dim_{cfg.skill_subgoal_cfg.visual_feature_dimension}/singletask_{cfg.multitask.training_task_id}"


    if cfg.meta.use_spatial_softmax:
        spatial_softmax_str = "_spatial_softmax"
    else:
        spatial_softmax_str = ""

    if cfg.meta_cvae_cfg.enable:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}_cvae"
    else:
        model_name = f"{output_dir}/meta_policy_{cfg.skill_subgoal_cfg.subgoal_type}_{cfg.meta_cvae_cfg.kl_coeff}_{cfg.meta_cvae_cfg.latent_dim}_False{spatial_softmax_str}"

    if cfg.multitask.testing_percentage < 1.0:
        model_name += f"_{cfg.multitask.testing_percentage}"
        
    if cfg.meta.random_affine:
        model_name += "_data_aug"

    summary_writer_name = model_name
    model_name += ".pth"
        
    return EasyDict({"output_dir": output_dir,
                     "model_name": model_name,
                     "summary_writer_name": summary_writer_name})
