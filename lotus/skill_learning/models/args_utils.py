import argparse
import json
from easydict import EasyDict
import pprint

def get_common_args(training=False):
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=""
    )

    parser.add_argument("--environment", type=str, default="PegInHoleEnv")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'")
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    )

    if training:
        parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
        )
        parser.add_argument(
            '--lr',
            type=float,
            default=1e-3,
        )

        parser.add_argument(
            '--min-lr',
            type=float,
            default=1e-4
        )

        parser.add_argument(
            '--num-workers',
            type=int,
            default=0,
        )

        parser.add_argument(
            '--num-epochs',
            type=int,
            default=100,
        )
    
    parser.add_argument(
        "--visualization",
        action="store_true",
    )
    parser.add_argument(
        "--video",
        action="store_true",
    )    
    parser.add_argument(
        "--use-goal",
        action="store_true",
    )
    parser.add_argument(
        "--use-embedding",
        action="store_true",
    )
    
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="single-policy",
        choices={"single-policy", "open-loop", "non-parametric", "parametric", "gt"},
        help="single-policy: no decomposition ; open-loop: open loop rollout chosen subskill"
    )

    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["agentview", "eye_in_hand", "force"]
    )

    parser.add_argument(
        '--subtask-id',
        type=int,
        nargs="+",
        default=[],
    )
    

    parser.add_argument(
        '--z-dim',
        type=int,
        default=128,
    )

    parser.add_argument(
        '--alpha-kl',
        type=float,
        default=0.05,
    )
    parser.add_argument(
        '--footprint',
        default="mean"
    )
    
    parser.add_argument(
        '--dist',
        default="l2",
        type=str,
    )

    parser.add_argument(
        '--segment-footprint',
        default="mean",
        type=str
    )

    parser.add_argument(
        '--K',
        default=4,
        type=int,
    )

    parser.add_argument(
        '--affinity',
        default="nearest_neighbors",
        type=str,
        choices={"nearest_neighbors", "rbf"}
    )

    parser.add_argument(
        '--no-skip',
        action='store_true'
    )

    parser.add_argument(
        '--use-gripper',
        action="store_true"
    )
    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
        '--eval-subtask-id',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--use-checkpoint',
        action="store_true"
    )

    parser.add_argument(
        '--use-image',
        action='store_true'
    )

    parser.add_argument(
        '--no-pooling',
        action='store_true'
    )

    parser.add_argument(
        '--no-eye-in-hand',
        action='store_true'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--agglomoration-step',
        type=int,
        default=10
    )

    args = parser.parse_args()
    return args

def get_goal_str(args):
    goal_str = ""
    if args.use_goal:
        goal_str = "goal_conditioned_"
    else:
        goal_str = ""

    if args.no_pooling:
        goal_str += "no_pooling_"
        
    if args.use_embedding:
        goal_str += "embedding_"

    if args.use_gripper:
        goal_str += "gripper_"
        
    if not args.use_eye_in_hand:
        goal_str += "no_wrist_"
    else:
        goal_str += "use_wrist_"

    if args.random_affine:
        goal_str += "data_aug_"
        
    return goal_str

def get_modalities_str(args):
    modalities = args.modalities
    modality_str = modalities[0]
    for modality in modalities[1:]:
        modality_str += f"_{modality}"
    modality_str += f"_{args.alpha_kl}"    
    if args.no_skip:
        modality_str += "_no_skip"

    return modality_str

def get_data_modality_str(args):
    """Return a str relating to the policy's data modality"""
    modality_str = args.data_modality[0]
    for modality in args.data_modality[1:]:
        modality_str += f"_{modality}"
    return modality_str

def update_json_config(args, debug=True):
    pp = pprint.PrettyPrinter(indent=4)
    cfg = EasyDict(vars(args))
    if args.config is not None:
        cfg_dict = json.load(open(args.config, "r"))
        cfg.update(cfg_dict)

    # process some args into config
    if cfg.no_eye_in_hand:
        cfg["use_eye_in_hand"] = not cfg.no_eye_in_hand
    if debug:
        pp.pprint(cfg)
    return cfg
    
