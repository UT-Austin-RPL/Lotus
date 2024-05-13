from easydict import EasyDict
import yaml

def get_goal_str(args):
    goal_str = ""
    # if args.skill_training.use_goal:
    #     goal_str = "goal_conditioned_"
    # else:
    #     goal_str = ""
    # if args.skill_training.no_pooling:
    #     goal_str += "no_pooling_"
        
    # if args.skill_training.use_embedding:
    #     goal_str += "embedding_"

    # if args.skill_training.use_gripper:
    #     goal_str += "gripper_"
        
    if not args.skill_training.use_eye_in_hand:
        goal_str += "no_wrist_"
    else:
        goal_str += "use_wrist_"

    if args.skill_training.random_affine:
        goal_str += "data_aug_"
    return goal_str


def get_modalities_str(args):
    modalities = args.repr.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{args.repr.alpha_kl}"    
    if args.repr.no_skip:
        modality_str += "_no_skip"

    return modality_str

def get_data_modality_str(args):
    """Return a str relating to the policy's data modality"""
    modality_str = args.skill_training.data_modality[0]
    for modality in args.skill_training.data_modality[1:]:
        modality_str += f"_{modality}"
    return modality_str
