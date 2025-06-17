import os
import yaml
import sys
import numpy as np
from IPython.core.splitinput import line_split
from scipy.spatial.transform import Rotation as R
import copy
from transforms3d.euler import euler2mat
import transforms3d

sys.path.append("../../RVT")
sys.path.append("/home/zk/Projects/3d_vla/RVT/rvt_our/")
print(sys.path)
import pickle
import torch
import cv2
import time

from multiprocessing import Value
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import rvt_our.mvt.config as default_mvt_cfg
import rvt_our.models.rvt_agent as rvt_agent
import rvt_our.config as default_exp_cfg

from rvt_our.utils.rvt_utils import load_agent as load_agent_state

from rvt_our.mvt.mvt import MVT
from rvt_our.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from botarm import Point

import open3d as o3d

def get_cam_extrinsic():
    trans = np.array([1.028818510131928, -0.04212360892289513, 0.6338377191806316])
    quat = np.array([-0.6333204911007358, -0.6400364927579377, 0.3240327100190967, 0.29027787777872616])  # x y z w

    transform = np.eye(4)
    rot = R.from_quat(quat)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = trans.T

    return transform


def vis_pcd(pcd, rgb):
    # 将点云和颜色转换为二维的形状 (N, 3)
    pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

    # 将点云和颜色信息保存为 PLY 文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
    pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
    # o3d.io.write_point_cloud(save_path, pcd)
    # 创建原点坐标系（size 可以根据需要设置）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])


def convert_pcd_to_base(
        type="3rd",
        pcd=[],
        extrinsic_matrix=None
):
    transform = extrinsic_matrix
    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)

    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


def load_agent(
        model_path=None,
        peract_official=False,
        peract_model_dir=None,
        exp_cfg_path=None,
        mvt_cfg_path=None,
        device=0,
        use_input_place_with_mean=False,
        palligemma_type=1,
        layer_index=-4,
        layer_concat=False):
    device = f"cuda:{device}"
    if not (peract_official):
        assert model_path is not None

        # load exp_cfg
        model_folder = os.path.join(os.path.dirname(model_path))

        exp_cfg = default_exp_cfg.get_cfg_defaults()
        if exp_cfg_path != None:
            exp_cfg.merge_from_file(exp_cfg_path)
        else:
            exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

        # NOTE: to not use place_with_mean in evaluation
        # needed for rvt-1 but not rvt-2
        if not use_input_place_with_mean:
            # for backward compatibility
            old_place_with_mean = exp_cfg.rvt.place_with_mean
            exp_cfg.rvt.place_with_mean = True
        exp_cfg.freeze()

        # create agent
        if exp_cfg.agent == "original":
            assert False
            # initialize PerceiverIO Transformer
            VOXEL_SIZES = [100]  # 100x100x100 voxels

            NUM_LATENTS = 512  # PerceiverIO latents
            BATCH_SIZE_TRAIN = 1
            perceiver_encoder = PerceiverIO(
                depth=6,
                iterations=1,
                voxel_size=VOXEL_SIZES[0],
                initial_dim=3 + 3 + 1 + 3,
                low_dim_size=4,
                layer=0,
                num_rotation_classes=72,
                num_grip_classes=2,
                num_collision_classes=2,
                num_latents=NUM_LATENTS,
                latent_dim=512,
                cross_heads=1,
                latent_heads=8,
                cross_dim_head=64,
                latent_dim_head=64,
                weight_tie_layers=False,
                activation="lrelu",
                input_dropout=0.1,
                attn_dropout=0.1,
                decoder_dropout=0.0,
                voxel_patch_size=5,
                voxel_patch_stride=5,
                final_dim=64,
            )

            # initialize PerceiverActor
            agent = PerceiverActorAgent(
                coordinate_bounds=SCENE_BOUNDS,
                perceiver_encoder=perceiver_encoder,
                camera_names=CAMERAS,
                batch_size=BATCH_SIZE_TRAIN,
                voxel_size=VOXEL_SIZES[0],
                voxel_feature_size=3,
                num_rotation_classes=72,
                rotation_resolution=5,
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                transform_augmentation=False,
                **exp_cfg.peract,
            )
        elif exp_cfg.agent == "our":
            mvt_cfg = default_mvt_cfg.get_cfg_defaults()
            if mvt_cfg_path != None:
                mvt_cfg.merge_from_file(mvt_cfg_path)
            else:
                mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
            mvt_cfg.add_proprio = False
            mvt_cfg.freeze()

            # for rvt-2 we do not change place_with_mean regardless of the arg
            # done this way to ensure backward compatibility and allow the
            # flexibility for rvt-1
            if mvt_cfg.stage_two:
                exp_cfg.defrost()
                exp_cfg.rvt.place_with_mean = old_place_with_mean
                exp_cfg.freeze()

            rvt = MVT(
                renderer_device=device,
                palligemma_type=palligemma_type,
                layer_concat=layer_concat,
                layer_index=layer_index,
                **mvt_cfg,
            )

            agent = rvt_agent.RVTAgent(
                network=rvt.to(device),
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                add_lang=mvt_cfg.add_lang,
                stage_two=mvt_cfg.stage_two,
                rot_ver=mvt_cfg.rot_ver,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
        else:
            raise NotImplementedError

        agent.build(training=False, device=device)
        load_agent_state(model_path, agent)
        agent.eval()

    elif peract_official:  # load official peract model, using the provided code
        assert False
        try:
            model_folder = os.path.join(os.path.abspath(peract_model_dir), "..", "..")
            train_cfg_path = os.path.join(model_folder, "config.yaml")
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        except FileNotFoundError:
            print("Config file not found, trying to load again in our format")
            train_cfg_path = "configs/peract_official_config.yaml"
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        agent.load_weights(peract_model_dir)
        agent.eval()

    print("Agent Information")
    print(agent)
    return agent


def vis_pcd_with_end_pred(pcd, rgb, end_pose=None, pred_pose=None, gt_pose=None):
    # 转换点云和颜色形状
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # 显示原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # -- 处理end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3])*0.001
    angles_deg_end = np.array(end_pose[3:6])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T_end)

    # -- 处理pred_pose
    if isinstance(pred_pose, str):
        pred_pose = [float(x) for x in pred_pose.strip('{}').split(',')]
    else:
        pred_pose = [float(x) for x in pred_pose]
    pos_pred = np.array(pred_pose[:3])*0.001
    angles_deg_pred = np.array(pred_pose[3:])
    angles_rad_pred = np.deg2rad(angles_deg_pred)
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
    T_pred = np.eye(4)
    T_pred[:3, :3] = rot_mat_pred
    T_pred[:3, 3] = pos_pred
    axis_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_pred.transform(T_pred)

    if gt_pose is not None:
        if isinstance(gt_pose, str):
            gt_pose = [float(x) for x in gt_pose.strip('{}').split(',')]
        else:
            gt_pose = [float(x) for x in gt_pose]
        pos_pred = np.array(gt_pose[:3])*0.001
        angles_deg_pred = np.array(gt_pose[3:6])
        angles_rad_pred = np.deg2rad(angles_deg_pred)
        rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
        T_pred = np.eye(4)
        T_pred[:3, :3] = rot_mat_pred
        T_pred[:3, 3] = pos_pred
        axis_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        axis_gt.transform(T_pred)

    # 显示所有内容
    if gt_pose is None:
        o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_pred, axis_end])
    else:
        o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_pred, axis_gt ,axis_end])


def get_pose(pose_dir, num):
    pose_path = os.path.join(pose_dir, f"{num}.pkl")
    with open(pose_path, "rb") as f:
        pose_data = pickle.load(f)
    pose_quat = pose_data[3:7]
    gripper_state = pose_data[-1]
    pose_eurl = transforms3d.euler.quat2euler(pose_quat, axes='sxyz')
    pose_eurl = np.rad2deg(np.asarray(pose_eurl))
    pose = [float(pose_data[0]*1000.), float(pose_data[1]*1000.), float(pose_data[2]*1000.), float(pose_eurl[0]), float(pose_eurl[1]), float(pose_eurl[2])]
    return pose,gripper_state


def _eval():
    base_path = "/home/zk/Projects/3d_vla/checkpoints/wbl_pipeline"
    model_path = os.path.join(base_path, "model_299_0606_r.pth")
    exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")
    mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
    episode_length = 30
    gripper_thres = 0.07
    cameras_view = ["3rd"]

    device = f"cuda:0"
    observation = {}


    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        device=0,
        palligemma_type=1,
        layer_index=-1,  # change it
        layer_concat=False,
    )



    data_path = "/home/zk/Projects/Datasets/put_block_on_the_plate_0604/7"  #
    pcd_dir = os.path.join(data_path, "3rd_cam_pcd")
    rgb_dir = os.path.join(data_path, "3rd_cam_rgb")
    pose_dir = os.path.join(data_path, "actions")
    extrinsic_matrix = os.path.join(data_path, "extrinsic_matrix.npy")
    instruction_path = os.path.join(data_path, "instruction.pkl")

    if not os.path.exists(extrinsic_matrix):
        extrinsic_matrix = get_cam_extrinsic()
    else:
        with open(extrinsic_matrix, 'rb') as f:
            extrinsic_matrix = np.load(f)
    print(extrinsic_matrix)

    if not os.path.exists(instruction_path):
        print("instruction file not exists")
        # instructions = [[["put the bottle in the microwave"]]]        #  放瓶子进微波炉
        # instructions = [[["put the cube in the top shelf"]]]          #  放方块在架子上
        instructions = [[["put the block on the plate"]]]             #  放方块进盘子
        # instructions = [[["press the bottle"]]]                       #  按压洗发水
        # instructions = [[["open the door"]]]                       #  开门

        # instructions = [[["put the lemonade in the microwave"]]]      #  金橘柠檬
        # instructions = [[["put the sprite in the microwave"]]]        #  雪碧
        # instructions = [[["put the green tea in the microwave"]]]     #  绿茶
        # instructions = [[["put the red tea in the microwave"]]]       #  冰红茶
        # instructions = [[["put the orange soda in the microwave"]]]     #  芬达

        # instructions = [[["put wolf in the drawer"]]]                 #  狼
        # instructions = [[["put giraffe in the drawer"]]]              #  长颈鹿
        # instructions = [[["put zebra in the drawer"]]]                #  斑马

    else:
        with open(instruction_path, 'rb') as f:
            instructions = [[[pickle.load(f)]]]

    observation["language_goal"] = instructions


    for i in range(5):
        pcd_path = os.path.join(pcd_dir, f"{i}.pkl")
        rgb_path = os.path.join(rgb_dir, f"{i}.pkl")
        pose_current, gripper_current = get_pose(pose_dir, i)
        if i < 4:
            pose_next, gripper_next = get_pose(pose_dir, i + 1)
        else:
            pose_next, gripper_next = pose_current, gripper_current
        with open(pcd_path, 'rb') as f:
            pcd_data = pickle.load(f)
        with open(rgb_path, 'rb') as f:
            rgb_data = pickle.load(f)

        pcd_data = pcd_data[:, :, :3]
        observation["3rd"] = {}
        observation["3rd"]["pcd"] = pcd_data
        rgb_data = rgb_data[:, :, :3]
        observation["3rd"]["rgb"] = rgb_data
        # pose_current = get_pose(pose_data, i + 1)
        # pose_next = get_pose(pose_data, i + 2)



        # 使用正则表达式匹配大括号中的内容
        claw_status = 1 if i == 2 or i == 3 else 0
        current_gripper = claw_status

        current_time = (1. - (i / float(episode_length - 1))) * 2. - 1.
        observation['low_dim_state'] = np.concatenate(
            [[current_gripper], [current_time]]).astype(np.float32)
        observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"], extrinsic_matrix)
        observation["3rd"]["pcd"] = observation["3rd"]["pcd"]
        end_pose = pose_current
        # vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])
        observation_origen = copy.deepcopy(observation)
        for key, v in observation.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k in ["rgb", "pcd"]:
                        v[sub_k] = np.transpose(v[sub_k], [2, 0, 1])
                        v[sub_k] = torch.from_numpy(v[sub_k]).to(device).unsqueeze(0).float().contiguous()
            elif isinstance(v, np.ndarray):
                observation[key] = torch.from_numpy(v).to(device).unsqueeze(0).contiguous()
        target_pos, target_quat_, target_gripper = agent.act_real(observation, cameras_view)

        # target_quat = [target_quat_[3], target_quat_[0], target_quat_[1], target_quat_[2]]
        target_quat = target_quat_
        if target_quat[0] < 0:
            # print("quat changed!")
            # print("before quat:",target_quat)
            target_quat = np.array(target_quat)
            target_quat = target_quat * (-1)
            # print("after quat:",target_quat)

        # x_range = (-0.222, 0.316)  # dobot
        # y_range = (-0.617, -0.20)
        # z_range = (0.06, 0.2)
        # target_pos[0] = np.clip(target_pos[0], x_range[0], x_range[1]) * 1000
        # target_pos[1] = np.clip(target_pos[1], y_range[0], y_range[1]) * 1000
        # target_pos[2] = np.clip(target_pos[2], z_range[0], z_range[1]) * 1000

        target_point = Point(target_pos*1000., target_quat, target_gripper)
        print("target_gripper", target_gripper)
        print("step:", i)
        print("groundtruth:", pose_next[:3])
        print("pred : ", target_pos*1000.)
        # print("Predicted target pos: ", target_pos, "Predicted target eurl: ", target_point.position_quaternion_claw,
        #       "Predicted target gripper: ", target_gripper)

        print("****************************")

        if target_gripper == 0:
            target_gripper = 1
        elif target_gripper == 1:
            target_gripper = 0
        else:
            assert False  # 训练时0开1关，测试时，1开0关

        vis_pcd_with_end_pred(observation_origen["3rd"]["pcd"], observation_origen["3rd"]["rgb"], end_pose,
                              target_point.position_quaternion_claw, gt_pose=pose_next)


if __name__ == '__main__':
    _eval()
