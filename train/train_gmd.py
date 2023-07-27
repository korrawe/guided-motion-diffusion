# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from configs import card

def main():
    # # TODO: remove flag.MODEL_ARCH
    # if flag.MODEL_ARCH == 'unet':
    #     if flag.TRAIN_MAX_LEN != 224:
    #         print('WARNING!: we always use 224 for UNET during training!')

    # Release version
    # args = train_args(base_cls=card.traj_unet_adagn_swx)
    args = train_args(base_cls=card.motion_abs_proj10_unet_adagn_xl)

    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_loss10)
    # args = train_args(base_cls=card.motion_abs_proj10_unet_adagn_xl_kps)
    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_kps)
    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_pose)
    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_kps_drop_redundant)
    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_drop_redundant)
    # args = train_args(base_cls=card.motion_abs_mdm_cond)
    
    # args = train_args(base_cls=card.motion_abs_mdm_cond_better)
    
    # args = train_args(base_cls=card.motion_abs_unet_adagn_xl_pose_drop_redundant)

    pprint(args.__dict__)
    # return

    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    # elif os.path.exists(args.save_dir) and not args.overwrite:
    #     raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type=args.augment_type,
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )

    data = get_dataset_loader(data_conf)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()