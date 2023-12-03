from argparse import ArgumentParser
import argparse
import os
import json
import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
    
    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--njoints", default = 47, type=int)

    # group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                     type=str,
                       help="Architecture types as reported in the paper.")

    group.add_argument("--mdm_condm", default='text', type=str,
                       help="Architecture types as reported in the paper.")

    group.add_argument("--audio_Q", action='store_true',
                       help="tran decoder audio Q, x KV")
    group.add_argument('--use_style', action='store_true', help="style control")
    group.add('--use_emo', action='store_true')
    group.add_argument('--fixtext', action='store_true', help="style control")

    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=1.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    # group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
    #                    help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--exp", type=str, default = 'test',
                       help="Path to save checkpoints and results.")
    group.add_argument("--save_dir", type=str, default = '/apdcephfs_cq2/share_1290939/yihaozhi/tx_a2g/MDM',
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='TensorboardPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=100, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=False, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def train_args():
    # parser = ArgumentParser()
    parser = configargparse.ArgParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    # add_ted_options(parser)
    add_beat_options(parser)

    return parser.parse_args()


def generate_args(use_motionclip=False):
    # parser = ArgumentParser()
    parser = configargparse.ArgParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    # add_ted_options(parser)
    add_beat_options(parser)
    if use_motionclip:
        add_motionclip(parser)
    return parse_and_load_from_model(parser)


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)


def add_beat_options(parser):
    group = parser.add_argument_group('beat')

    group.add("-c", "--config", required=True, is_config_file=True)
    # save the objective score
    group.add("--csv", default="0118_git1.csv", type=str)
    group.add("--trainer", default="camn", type=str)
    parser.add("--new_cache", action='store_true')
    parser.add('--use_sem', action='store_true')

    parser.add("--speakers", action='append')

    # group.add("--exp", default="test", type=str)

    # ------------- path and save name ---------------- #
    group.add("--is_train", default=True, type=str2bool)
    # different between environments
    group.add("--root_path", default="")
    group.add("--out_root_path", default="/outputs/audio2pose/", type=str)
    group.add("--train_data_path", default="/datasets/trinity/train/", type=str)
    group.add("--val_data_path", default="/datasets/trinity/val/", type=str)
    group.add("--test_data_path", default="/datasets/trinity/test/", type=str)
    group.add("--mean_pose_path", default="/datasets/trinity/train/", type=str)
    group.add("--std_pose_path", default="/datasets/trinity/train/", type=str)
    group.add("--vocab_path", default="/datasets/trinity/train/", type=str)

    # for pretrian weights
    group.add("--torch_hub_path", default="../../datasets/checkpoints/", type=str)

    # pretrained vae for evaluation
    # load vae name = eval_model_type_vae_length
    group.add("--model_name_last", default="last.pth", type=str)
    group.add("--model_name_best", default="best.pth", type=str)
    group.add("--eval_model", default="motion_autoencoder", type=str)
    group.add("--e_name", default="HalfEmbeddingNet", type=str)
    group.add("--e_path", default="/datasets/beat/generated_data/self_vae_128.bin")
    group.add("--variational_encoding", default=False, type=str2bool) 
    group.add("--vae_length", default=256, type=int)

    # --------------- data ---------------------------- #
    group.add("--dataset", default="beat", type=str)
    group.add("--use_aug", default=False, type=str2bool)
    group.add("--disable_filtering", default=False, type=str2bool)
    group.add("--clean_first_seconds", default=0, type=int)
    group.add("--clean_final_seconds", default=0, type=int)

    group.add("--audio_rep", default="wave16k", type=str)
    group.add("--word_rep", default="None", type=str)
    group.add("--emo_rep", default="None", type=str)
    group.add("--sem_rep", default="None", type=str)
    group.add("--audio_fps", default=16000, type=int)
    #parser.add("--audio_dims", default=1, type=int)
    group.add("--facial_rep", default="facial39", type=str)
    group.add("--facial_fps", default=15, type=int)
    group.add("--facial_dims", default=39, type=int)
    group.add("--pose_rep", default="fps15_trinity_rot_123", type=str)
    group.add("--pose_fps", default=15, type=int)
    group.add("--pose_dims", default=123, type=int)
    group.add("--speaker_id", default=False, type=str2bool)
    group.add("--audio_norm", default=False, type=str2bool)
    
    group.add("--pose_length", default=34, type=int)
    group.add("--pre_frames", default=4, type=int)
    group.add("--stride", default=10, type=int)
    group.add("--pre_type", default="zero", type=str)
    
    
    # --------------- model ---------------------------- #
    group.add("--pretrain", default=False, type=str2bool)
    group.add("--model", default="camn", type=str)
    group.add("--g_name", default="CaMN", type=str)
    group.add("--d_name", default="ConvDiscriminator", type=str)
    group.add("--dropout_prob", default=0.3, type=float)
    group.add("--n_layer", default=4, type=int)
    group.add("--hidden_size", default=300, type=int)
    group.add("--audio_f", default=128, type=int)
    group.add("--facial_f", default=128, type=int)
    group.add("--speaker_f", default=0, type=int)
    group.add("--word_f", default=0, type=int)
    group.add("--emotion_f", default=0, type=int)
    # Self-designed "Multi-Stage", "Seprate", or "Original"
    group.add("--finger_net", default="original", type=str)
    
    
    # --------------- training ------------------------- #
    group.add("--epochs", default=401, type=int)
    group.add("--no_adv_epochs", default=4, type=int)
    group.add('-b',"--batch_size", default=512, type=int)
    group.add("--opt", default="adam", type=str)
    group.add("--lr_base", default=0.00025, type=float)
    # group.add("--weight_decay", default=0., type=float)
    # for warmup and cosine
    group.add("--lr_min", default=1e-7, type=float)
    # for sgd
    group.add("--momentum", default=0.8, type=float)
    group.add("--nesterov", default=True, type=str2bool)
    # for adam
    group.add("--opt_betas", default=[0.5, 0.999], type=list)
    group.add("--amsgrad", default=False, type=str2bool)
    group.add("--lr_policy", default="none", type=str)
    group.add("--d_lr_weight", default=0.2, type=float)
    group.add("--rec_weight", default=500, type=float)
    group.add("--adv_weight", default=20.0, type=float)
    group.add("--fid_weight", default=0.0, type=float)
    group.add("--vel_weight", default=0.0, type=float)
    group.add("--acc_weight", default=0.0, type=float)
#    parser.add("--gan_noise_size", default=0, type=int)

    # --------------- device -------------------------- #
    group.add("--random_seed", default=2021, type=int)
    group.add("--deterministic", default=True, type=str2bool)
    group.add("--benchmark", default=True, type=str2bool)
    group.add("--cudnn_enabled", default=True, type=str2bool)
    # mix precision
    group.add("--apex", default=False, type=str2bool)
    group.add("--gpus", default=[0], type=list)
    group.add("--loader_workers", default=0, type=int)
    # logging
    group.add("--log_period", default=10, type=int)
    group.add("--test_period", default=20, type=int)    
    group.add_argument("-w", "--num_workers", type=int, default=16, help="dataloader worker size")


def add_motionclip(parser):
    group = parser.add_argument_group('motionclip')
    group.add_argument('--N_CTX', type=int, default=4, help='')
    group.add_argument('--CLASS_TOKEN_POSITION', type=str, default="end", help='Which device the training is on')
    group.add_argument("--LearnablePrompt", type = bool,  default= False)
    group.add_argument("--lam_cos_loss", type = float, default=0.001)
    group.add_argument("--lam_inter_cos_loss", type = float, default=0.0)
    group.add_argument("--test_only", type = bool, default= False)
    group.add_argument("--use_bert", type = bool, default= False)
    group.add_argument("--use_bert_feature", type = str, default= "sos")
    group.add_argument("--n_pre_poses", type=int, default=4, help="n_pre_poses")
    group.add_argument('--use_reparam', default= False, type=bool, help="reparam")
    # group.add_argument('--use_style', default= False, type=bool, help="style")
    group.add_argument('--autoregressive', default= False, type=bool, help="ar")
    # group.add_argument('--audio_model', default= "default", type=str, help="produce cond motion")