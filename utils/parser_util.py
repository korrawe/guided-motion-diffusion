from argparse import ArgumentParser
import argparse
from dataclasses import dataclass, field, fields
import os
import json
from typing import Tuple, Union
from utils.hfargparse import HfArgumentParser


@dataclass
class BaseOptions:
    cuda: bool = field(
        default=True, metadata={"help": "Use cuda device, otherwise use CPU."})
    device: int = field(default=0, metadata={"help": "Device id to use."})
    seed: int = field(default=10, metadata={"help": "For fixing random seed."})


@dataclass
class DiffusionOptions:
    noise_schedule: str = field(default='cosine',
                                metadata={"help": "Noise schedule type"})
    diffusion_steps: int = field(
        default=1000,
        metadata={
            "help": "Number of diffusion steps (denoted T in the paper)"
        })
    sigma_small: bool = field(default=True,
                              metadata={"help": "Use smaller sigma values."})
    predict_xstart: bool = field(default=True,
                                 metadata={"help": "Predict xstart."})
    use_ddim: bool = field(default=False, metadata={"help": "Use ddim."})
    clip_range: float = field(default=6.0, metadata={"help": "Clip range."})


@dataclass
class ModelOptions:
    arch: str = field(
        default='trans_enc',
        metadata={"help": "Architecture types as reported in the paper."})
    emb_trans_dec: bool = field(
        default=False,
        metadata={
            "help":
            "For trans_dec architecture only, if true, will inject condition as a class token (in addition to cross-attention)."
        })
    layers: int = field(default=8, metadata={"help": "Number of layers."})
    latent_dim: int = field(default=512,
                            metadata={"help": "Transformer/GRU width."})
    ff_size: int = field(default=1024,
                         metadata={"help": "Transformer feedforward size."})
    dim_mults: Tuple[float] = field(
        default=(2, 2, 2, 2), metadata={"help": "Unet channel multipliers."})
    unet_adagn: bool = field(
        default=True, metadata={"help": "Unet adaptive group normalization."})
    unet_zero: bool = field(
        default=True, metadata={"help": "Unet zero weight initialization."})
    out_mult: bool = field(
        default=1,
        metadata={"help": "UNET/TF large variation's feature multiplier."})
    cond_mask_prob: float = field(
        default=.1,
        metadata={
            "help":
            "The probability of masking the condition during training. For classifier-free guidance learning."
        })
    lambda_rcxyz: float = field(default=0.0,
                                metadata={"help": "Joint positions loss."})
    lambda_vel: float = field(default=0.0,
                              metadata={"help": "Joint velocity loss."})
    lambda_fc: float = field(default=0.0,
                             metadata={"help": "Foot contact loss."})
    unconstrained: bool = field(
        default=False,
        metadata={
            "help":
            "Model is trained unconditionally. That is, it is constrained by neither text nor action. Currently tested on HumanAct12 only."
        })


@dataclass
class DataOptions:
    dataset: str = field(default='humanml',
                         metadata={
                             "help": "Dataset name (choose from list).",
                             "choices":
                             ['humanml', 'kit', 'humanact12', 'uestc']
                         })
    data_dir: str = field(
        default="",
        metadata={
            "help":
            "If empty, will use defaults according to the specified dataset."
        })
    abs_3d: bool = field(default=False, metadata={"help": "Use absolute 3D."})
    traj_only: bool = field(default=False,
                            metadata={"help": "Use trajectory model."})
    xz_only: Tuple[int] = field(
        default=False,
        metadata={"help": "trajectory model with 2 input channels."})
    use_random_proj: bool = field(default=False,
                                  metadata={"help": "Use random projection."})
    random_proj_scale: float = field(
        default=10.0, metadata={"help": "Random projection scale."})
    augment_type: str = field(default='none',
                              metadata={
                                  "help": "Augmentation type.",
                                  "choices": ['none', 'rot', 'full']
                              })
    std_scale_shift: Tuple[float] = field(
        default=(1.0, 0.0),
        metadata={"help": "Adjusting the std by scale and shift."})
    drop_redundant: bool = field(
        default=False,
        metadata={"help": "Drop redundant joints information for HumanML3D only. "
                  "Keep only 4 (root) + 21*3 joints position"})


@dataclass
class TrainingOptions:
    save_dir: str = field(
        default=None,
        metadata={"help": "Path to save checkpoints and results."})
    overwrite: bool = field(
        default=False,
        metadata={
            "help": "If True, will enable to use an already existing save_dir."
        })
    batch_size: int = field(default=64,
                            metadata={"help": "Batch size during training."})
    train_platform_type: str = field(
        default='TensorboardPlatform',
        metadata={
            "help":
            "Choose platform to log results. NoPlatform means no logging.",
            "choices":
            ['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform']
        })
    lr: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.,
                                metadata={"help": "Optimizer weight decay."})
    grad_clip: float = field(default=0, metadata={"help": "Gradient clip."})
    use_fp16: bool = field(default=False, metadata={"help": "Use fp16."})
    avg_model_beta: float = field(
        default=0, metadata={"help": "Average model beta; 0 = disabled."})
    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2."})
    lr_anneal_steps: int = field(
        default=0, metadata={"help": "Number of learning rate anneal steps."})
    eval_batch_size: int = field(
        default=32,
        metadata={
            "help":
            "Batch size during evaluation loop. Do not change this unless you know what you are doing. T2m precision calculation is based on fixed batch size 32."
        })
    eval_split: str = field(default='test',
                            metadata={
                                "help":
                                "Which split to evaluate on during training.",
                                "choices": ['val', 'test']
                            })
    eval_during_training: bool = field(
        default=False,
        metadata={"help": "If True, will run evaluation during training."})
    eval_rep_times: int = field(
        default=3,
        metadata={
            "help":
            "Number of repetitions for evaluation loop during training."
        })
    eval_num_samples: int = field(
        default=1_000,
        metadata={
            "help": "If -1, will use all samples in the specified split."
        })
    log_interval: int = field(default=1_000,
                              metadata={"help": "Log losses each N steps"})
    save_interval: int = field(
        default=100_000,
        metadata={"help": "Save checkpoints and run evaluation each N steps"})
    num_steps: int = field(
        default=1_200_000,
        metadata={
            "help": "Training will stop after the specified number of steps."
        })
    num_frames: int = field(
        default=60,
        metadata={
            "help":
            "Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored."
        })
    resume_checkpoint: str = field(
        default="",
        metadata={
            "help":
            "If not empty, will start from the specified checkpoint (path to model###.pt file)."
        })
    apply_zero_mask: bool = field(default=False,
                                  metadata={"help": "Apply zero mask."})
    traj_extra_weight: float = field(
        default=1.0, metadata={"help": "Trajectory extra weight."})
    time_weighted_loss: bool = field(default=False,
                                     metadata={"help": "Time weighted loss."})
    train_x0_as_eps: bool = field(default=False,
                                  metadata={"help": "Train x0 as eps."})


@dataclass
class SamplingOptions:
    model_path: str = field(
        default='',
        metadata={"help": "Path to model####.pt file to be sampled."})
    output_dir: str = field(
        default='',
        metadata={
            "help":
            "Path to results dir (auto created by the script). If empty, will create dir in parallel to checkpoint."
        })
    num_samples: int = field(
        default=10,
        metadata={
            "help":
            "Maximal number of prompts to sample, if loading dataset from file, this field will be ignored."
        })
    num_repetitions: int = field(
        default=3,
        metadata={
            "help": "Number of repetitions, per sample (text prompt/action)"
        })
    guidance_param: float = field(
        default=2.5,
        metadata={
            "help":
            "For classifier-free sampling - specifies the s parameter, as defined in the paper."
        })


@dataclass
class GenerateOptions:
    motion_length: float = field(
        default=11.2,
        metadata={
            "help":
            "The length of the sampled motion [in seconds]. Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)."
        })
    motion_length_cut: float = field(
        default=6.0,
        metadata={
            "help":
            "The actual length of the sampled motion [in seconds]. This is necessary for UNET because it always generate with maximum length."
        })
    input_text: str = field(
        default='',
        metadata={
            "help":
            "Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset."
        })
    action_file: str = field(
        default='',
        metadata={
            "help":
            "Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. If no file is specified, will take action names from dataset."
        })
    text_prompt: str = field(
        default='',
        metadata={
            "help":
            "A text prompt to be generated. If empty, will take text prompts from dataset."
        })
    action_name: str = field(
        default='',
        metadata={
            "help":
            "An action name to be generated. If empty, will take text prompts from dataset."
        })
    guidance_mode: str = field(
        default='no',
        metadata={
            "help":
            "Select the guideance mode for generataion. Possible options are [no, mdm_legacy, trajectory, kps, sdf].",
            "choices": ['no', 'mdm_legacy', 'trajectory', 'kps', 'sdf']
        })
    classifier_scale: float = field(
        default=100.0,
        metadata={
            "help":
            "A scaling factor for the gradient from the classifier. Use the same scale for both model in two-staged case"
        })
    do_inpaint: bool = field(
        default=False, metadata={"help": "If True, will perform inpainting."})
    # gen_avg_model: bool = field(
    #     default=True, metadata={"help": "If True, use average model for generation."})
    gen_reward_model: bool = field(
        default=False, metadata={"help": "If True, use an eps model to propagate the loss gradient for the xstart model."})
    gen_two_stages: bool = field(
        default=False, metadata={"help": "If True, generate trajectory first, then use it to impute to generate motion."})
    gen_mse_loss: bool = field(
        default=True, metadata={"help": "If True, use MSE loss for classifier. Otherwise, use L1 loss"})
    p2p_impute: bool = field(
        default=True, metadata={"help": "If True, use point-to-point guidance for trajectory."})
    interactive: bool = field(
        default=False, 
        metadata={
            "help": "If True, use interactive mode for selecting key locations. Override other options when selecting conditioning pattern."
        })


@dataclass
class EditOptions:
    edit_mode: str = field(
        default='in_between',
        metadata={
            "help":
            "Defines which parts of the input motion will be edited.\n(1) in_between - suffix and prefix motion taken from input motion, middle motion is generated.\n(2) upper_body - lower body joints taken from input motion, upper body is generated."
        })
    text_condition: str = field(
        default='',
        metadata={
            "help":
            "Editing will be conditioned on this text prompt. If empty, will perform unconditioned editing."
        })
    prefix_end: float = field(
        default=0.25,
        metadata={
            "help":
            "For in_between editing - Defines the end of input prefix (ratio from all frames)."
        })
    suffix_start: float = field(
        default=0.75,
        metadata={
            "help":
            "For in_between editing - Defines the start of input suffix (ratio from all frames)."
        })


@dataclass
class EvaluationOptions:
    model_path: str = field(
        default='',
        metadata={"help": "Path to model####.pt file to be sampled."})
    eval_mode: str = field(
        default='wo_mm',
        metadata={
            "help":
            "wo_mm (t2m only) - 20 repetitions without multi-modality metric; mm_short (t2m only) - 5 repetitions with multi-modality metric; debug - short run, less accurate results.full (a2m only) - 20 repetitions."
        })
    guidance_param: float = field(
        default=2.5,
        metadata={
            "help":
            "For classifier-free sampling - specifies the s parameter, as defined in the paper."
        })
    impute_until: int = field(default=None, metadata={"help": "impute until"})
    skip_first: int = field(default=None,
                            metadata={"help": "skip first stage"})
    eval_use_avg: bool = field(
        default=True,
        metadata={
            "help":
            "Use average model for sampling. Otherwise, use the last model."
        })
    full_traj_inpaint: bool = field(
        default=False,
        metadata={
            "help":
            "Condition on ground truth trajectory for inpainting."
        })


@dataclass
class FullModelArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                   TrainingOptions, SamplingOptions, GenerateOptions):
    pass


@dataclass
class TrainArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                TrainingOptions):
    pass


def train_args(base_cls=TrainArgs):
    """
    Args:
        base_cls: The class to use as the base class for the parser. 
            You can provide a different set of default values by subclassing TrainArgs.
    """
    parser = HfArgumentParser(base_cls)
    args: TrainArgs = parser.parse_args_into_dataclasses()[0]
    return args


@dataclass
class GenerateArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                   TrainingOptions, SamplingOptions, GenerateOptions):
    pass


def generate_args(model_path=None) -> GenerateArgs:
    parser = HfArgumentParser(GenerateArgs)
    args = parse_and_load_from_model(parser, model_path)
    return args


def edit_args():
    raise NotImplementedError()
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


@dataclass
class EvalArgs(BaseOptions, DataOptions, ModelOptions, DiffusionOptions,
                   TrainingOptions, SamplingOptions, GenerateOptions, EvaluationOptions):
    pass


def eval_args(model_path=None) -> EvalArgs:
    parser = HfArgumentParser(EvalArgs)
    args = parse_and_load_from_model(parser, model_path)
    return args


def evaluation_parser(trajectory_model=False, traj_model_path=None):
    # if trajectory_model:
    #     # Overwrite flag with flag_traj
    #     import sys
    #     import flag_traj as flag

    parser = HfArgumentParser([BaseOptions, EvaluationOptions])
    # parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    # add_base_options(parser)
    # add_evaluation_options(parser)
    return parse_and_load_from_model(parser, traj_model_path)


def parse_and_load_from_model(parser: HfArgumentParser,
                              model_path=None):
    ''' If model_path is given as parameters, it will override the cmd input.
    '''
    args: FullModelArgs = parser.parse_args()

    args_to_overwrite = []
    for cls in [DataOptions, ModelOptions, DiffusionOptions, TrainingOptions]: # EvaluationOptions
        for field in fields(cls):
            args_to_overwrite.append(field.name)

    # load args from model
    if model_path is not None:
        print(" - model_path is given in the function call. Cmd path, if any, will be ignored.")
        args.model_path = model_path
    else:
        model_path = args.model_path

    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file was not found! {args_path}'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            # print(f'setting {a} to {model_args[a]}')
            setattr(args, a, model_args[a])
        elif 'cond_mode' in model_args:  # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)
        else:
            print(
                'Warning: was not able to load [{}], using default value [{}] instead.'
                .format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args
