from utils.parser_util import *
from dataclasses import dataclass
from configs import data, model

###########################
# MOTION MODELS
###########################


@dataclass
class motion_rel_mdm(
        data.humanml_motion_rel,
        model.motion_mdm,
):
    save_dir: str = 'save/my_humanml_trans_enc_512_test'


@dataclass
class motion_abs_mdm(
        data.humanml_motion_abs,
        model.motion_mdm,
):
    save_dir: str = 'save/my_abs3d_2'


@dataclass
class motion_abs_mdm_proj1(
        data.humanml_motion_proj1,
        model.motion_mdm,
):
    save_dir: str = 'save/my_abs3d_proj_1'


@dataclass
class motion_abs_mdm_proj2(
        data.humanml_motion_proj2,
        model.motion_mdm,
):
    save_dir: str = 'save/my_abs3d_proj_2'


@dataclass
class motion_abs_mdm_proj5(
        data.humanml_motion_proj5,
        model.motion_mdm,
):
    save_dir: str = 'save/my_abs3d_proj_5'


@dataclass
class motion_abs_mdm_proj10(
        data.humanml_motion_proj10,
        model.motion_mdm,
):
    save_dir: str = 'save/my_abs3d_proj_10_2'


@dataclass
class motion_rel_unet_adagn_xl(
        data.humanml_motion_rel,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_rel_loss1_fp16_clipwd_224'


###########################
# UNET XL
###########################


@dataclass
class motion_abs_unet_adagn_xl(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_loss1_fp16_clipwd_224'


@dataclass
class motion_abs_unet_adagn_xl_loss2(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl_loss2,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_loss2_fp16_clipwd_224'


@dataclass
class motion_abs_unet_adagn_xl_loss5(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl_loss5,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_loss5_fp16_clipwd_224'


@dataclass
class motion_abs_unet_adagn_xl_loss10(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl_loss10,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_loss10_fp16_clipwd_224'


###########################
# UNET XL + PROJ
###########################


@dataclass
class motion_abs_proj1_unet_adagn_xl(
        data.humanml_motion_proj1,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_proj1_fp16_clipwd_224'


@dataclass
class motion_abs_proj2_unet_adagn_xl(
        data.humanml_motion_proj2,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_proj2_fp16_clipwd_224'


@dataclass
class motion_abs_proj5_unet_adagn_xl(
        data.humanml_motion_proj5,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_proj5_fp16_clipwd_224'


@dataclass
class motion_abs_proj10_unet_adagn_xl(
        data.humanml_motion_proj10,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224'


###########################
# UNET XL - DROP REDUNDANT
###########################
@dataclass
class motion_abs_unet_adagn_xl_drop_redundant(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_224_drop_redundant'
    drop_redundant: bool = True
    train_keypoint_mask: str = 'no' # 'keypoints'

###########################
# UNET XL + CFG KEYPOINT
###########################
@dataclass
class motion_abs_unet_adagn_xl_kps(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_fp16_clipwd_224_kps_mi3' # _long_fix' # all'
#     resume_checkpoint: str = 'save/unet_adazero_xl_x0_abs_fp16_clipwd_224_kps_long_fix_15/model001700000.pt'
    train_keypoint_mask: str = 'keypoints'

###########################
# UNET XL + CFG KEYPOINT - DROP REDUNDANT
###########################
@dataclass
class motion_abs_unet_adagn_xl_kps_drop_redundant(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_224_kps_drop_redundant_seed11' # all'
    drop_redundant: bool = True
    train_keypoint_mask: str = 'keypoints'
    seed: int = 11


###########################
# UNET XL + CFG KEY *POSE*
###########################
@dataclass
class motion_abs_unet_adagn_xl_pose(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_pose'
    train_keypoint_mask: str = 'keyposes'


###########################
# UNET XL + CFG KEY *POSE* - DROP REDUNDANT
###########################
@dataclass
class motion_abs_unet_adagn_xl_pose_drop_redundant(
        data.humanml_motion_abs,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_pose_drop_redundant'
    train_keypoint_mask: str = 'keyposes'
    drop_redundant: bool = True


###########################
# UNET XL + PROJ + KEYPOINT
###########################
@dataclass
class motion_abs_proj10_unet_adagn_xl_kps(
        data.humanml_motion_proj10,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224_kps'


###########################
# TRAJ MODELS
###########################


@dataclass
class traj_unet_adagn_swx(
        data.humanml_traj,
        model.traj_unet_adagn_swx,
):
    save_dir: str = 'save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224'


@dataclass
class traj_unet_xxs(
        data.humanml_traj,
        model.traj_unet_xxs,
):
    save_dir: str = 'save/traj_unet_xxs_eps_abs_fp16_clipwd_224'


###########################
# TRAJ MODELS + KEYPOINT
###########################
@dataclass
class traj_unet_adagn_swx_kps(
        data.humanml_traj,
        model.traj_unet_adagn_swx,
):
    save_dir: str = 'save/traj_unet_adazero_swxs_eps_abs_fp16_clipwd_224_traj'      

###########################
# TRANSFORMER MODELS + CONDITION
###########################
@dataclass
class motion_abs_mdm_cond(
        data.humanml_motion_abs,
        model.motion_mdm,
):
    save_dir: str = 'save/trans_abs_kps'
    train_keypoint_mask: str = 'keypoints'

###########################
# TRANSFORMER MODELS + BETTER KPS CONDITION
###########################
@dataclass
class motion_abs_mdm_cond_better(
        data.humanml_motion_abs,
        model.motion_mdm,
):
    save_dir: str = 'save/trans_abs_kps_better'
    train_keypoint_mask: str = 'keypoints_better_cond'

###########################
# TRANSFORMER MODELS + *POSE* CONDITION
###########################
@dataclass
class motion_abs_mdm_pose_cond(
        data.humanml_motion_abs,
        model.motion_mdm,
):
    save_dir: str = 'save/trans_abs_poses_naive'
    train_keypoint_mask: str = 'keyposes'
