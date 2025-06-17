# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import yaml
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from frankapy import FrankaArm
from autolab_core import RigidTransform
sys.path.append("/home/guangyun/Project/FiveAges/3d_vla_code/RVT")
from data_collection_frankapy import osc_move,gripper_open_flag
from utils.real_camera_utils_new import Camera,get_cam_extrinsic


import torch
import cv2
import time

from multiprocessing import Value
# from tensorflow.python.summary.summary_iterator import summary_iterator
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

def convert_pcd_to_base(
            type="3rd",
            pcd=[]
        ):
        transform = get_cam_extrinsic(type)
        
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



def _eval():

    base_path = "/media/casia/data2/lpy/3D_VLA/code/checkpoints/new_pipeline"
    # base_path = "/data2/lpy/3D_VLA/code/checkpoints/8e5_3rd_new_new"
    # model_path = os.path.join(base_path, "debug_800.pth") 
    # model_path = os.path.join(base_path, "rlbench_pretrain_model_10_300_debug_color.pth") 
    model_path = os.path.join(base_path, "rlbench_front_with_proprio_80_no_finetune_real_smallsize.pth")  # change 1
    exp_cfg_path = os.path.join(base_path, "exp_cfg.yaml")  
    mvt_cfg_path = os.path.join(base_path, "mvt_cfg.yaml")
    episode_length = 3 # change 2
    gripper_thres=0.07
    # cameras_view=["3rd", "wrist"]
    cameras_view=["3rd"] # change 3
    
    device = f"cuda:0"
    observation = {}
    observation['tasks'] = "press sanitizer"
    # generalization  settings

    # change 4


    # instructions = [[["place the red block in the green plate close to black shirt"]]]
    # instructions = [[["open the top drawer"]]]
    # instructions = [[["place the red block on the black bag"]]]
    # instructions = [[["place the red block in the purple plate"]]]
    # instructions = [[["place the red block in the left plate"]]]
    # instructions = [[["place the red block on the labtop"]]]
    # instructions = [[["place the blue block on the red block"]]]
    # instructions = [[["place the green block in top shelf"]]]
    # instructions = [[["place the red block on the green block"]]]
    # instructions = [[["place the blue block on the red block"]]]
    # instructions = [[["place the yellow block on the red block"]]]
    # instructions = [[["place the orange block on the red block"]]]
    # instructions = [[["place the red block in the green plate"]]]
    # instructions = [[["place the green block in the blue plate"]]]
    # instructions = [[["place the blue block in the purple plate"]]]
    # instructions = [[["place the blue block in top shelf"]]]
    # instructions = [[["place the red apple in the green plate"]]]
    # instructions = [[["place the apple in the green plate"]]]
    # instructions = [[["place the red block in the green plate close to the apple"]]]
    # instructions = [[["place the bottle in the blue plate"]]]
    instructions = [[["slide the block to blue target"]]]
    # instructions = [[["press the green button"]]]


    # instructions = [[["place the green block on the red block"]]]
    # instructions = [[["place the green block in bottom shelf"]]]
    # observation['tasks'] = "press sanitizer"
    # instructions = [[["press sanitizer"]]]
    # instructions = [[["press the bottle"]]]
    # instructions = [[["place the red block on the blue block"]]]
    # observation['tasks'] = "put item in drawer"
    # instructions = [[["put the blue block in the drawer"]]]
    cameras = Camera(camera_type="3rd") # modify it
    time.sleep(1)
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    time.sleep(2)
    agent = load_agent(
        model_path=model_path,
        exp_cfg_path=exp_cfg_path,
        mvt_cfg_path=mvt_cfg_path,
        device=0,
        palligemma_type=1,
        layer_index=-4,
        layer_concat=False,
        )

    observation["language_goal"]=instructions

    for step in range(episode_length-1):
        camera_info = cameras.capture()
        observation["3rd"] = camera_info["3rd"]
        # observation["wrist"] = camera_info["wrist"]
        observation["3rd"]["rgb"]=observation["3rd"]["rgb"][:,:,::-1].copy()
        # observation["wrist"]["rgb"]=observation["wrist"]["rgb"][:,:,::-1].copy()  # bgr2rgb
        # import pickle
        # idx = 0
        # step = 0
        # observation["3rd"] = {}
        # observation["wrist"] = {}
        # with open(f"/data2/cyx/realworld_datasets/place_block_in_plate/{idx}/3rd_cam_rgb/{step}.pkl", 'rb') as f:
        #     observation["3rd"]["rgb"] = pickle.load(f)
        # with open(f"/data2/cyx/realworld_datasets/place_block_in_plate/{idx}/3rd_cam_pcd/{step}.pkl", 'rb') as f:
        #     observation["3rd"]["pcd"] = pickle.load(f)
        # with open(f"/data2/cyx/realworld_datasets/place_block_in_plate/{idx}/wrist_cam_rgb/{step}.pkl", 'rb') as f:
        #     observation["wrist"]["rgb"] = pickle.load(f)
        # with open(f"/data2/cyx/realworld_datasets/place_block_in_plate/{idx}/wrist_cam_pcd/{step}.pkl", 'rb') as f:
        #     observation["wrist"]["pcd"] = pickle.load(f)
        
        current_gripper = gripper_open_flag(fa,gripper_thres) # 1 开 0关
        
        current_time = (1. - (step / float(episode_length - 1))) * 2. - 1.
        observation['low_dim_state'] = np.concatenate(
            [[current_gripper], [current_time]]).astype(np.float32)
        
        observation["3rd"]["pcd"] = convert_pcd_to_base("3rd", observation["3rd"]["pcd"])
        # observation["wrist"]["pcd"] = convert_pcd_to_base("wrist", observation["wrist"]["pcd"])
        
        import open3d as o3d
        def vis_pcd(pcd, rgb):

            # 将点云和颜色转换为二维的形状 (N, 3)
            pcd_flat = pcd.reshape(-1, 3)  # (200 * 200, 3)
            rgb_flat = rgb.reshape(-1, 3) / 255.0  # (200 * 200, 3)

            # 将点云和颜色信息保存为 PLY 文件
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_flat)  # 设置点云位置
            pcd.colors = o3d.utility.Vector3dVector(rgb_flat)  # 设置对应的颜色
            # o3d.io.write_point_cloud(save_path, pcd)
            o3d.visualization.draw_geometries([pcd])
        # test_pcd = np.concatenate((observation["3rd"]["pcd"], observation["wrist"]["pcd"]), axis=0)
        # test_rgb = np.concatenate((observation["3rd"]["rgb"], observation["wrist"]["rgb"]), axis=0)
        # vis_pcd(test_pcd, test_rgb)
        # vis_pcd(observation["3rd"]["pcd"], observation["3rd"]["rgb"])
                
        for key, v in observation.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if sub_k in ["rgb", "pcd"]:
                        v[sub_k] = np.transpose(v[sub_k], [2, 0, 1]) 
                        v[sub_k] = torch.from_numpy(v[sub_k]).to(device).unsqueeze(0).float().contiguous()
                    
            elif isinstance(v, np.ndarray):
                observation[key] = torch.from_numpy(v).to(device).unsqueeze(0).contiguous()
                
        target_pos, target_quat, target_gripper = agent.act_real(observation,cameras_view)
        # if target_quat[3] <0 :
        #     print("quat changed!!!!")
        #     target_quat=-target_quat
        # 这里得到的quat为x y z w格式，需要转换为w x y z格式送入到osc move当中
        target_quat=[target_quat[3],target_quat[0],target_quat[1],target_quat[2]]
        
        x_range = (0, 0.8)
        y_range = (-0.4, 0.4)
        z_range = (-0.05, 0.55)
        
        target_pos[0] = np.clip(target_pos[0], x_range[0], x_range[1])
        target_pos[1] = np.clip(target_pos[1], y_range[0], y_range[1])
        target_pos[2] = np.clip(target_pos[2], z_range[0], z_range[1])
        
        if target_pos[-1] < 0.01:
            target_pos[-1] = 0.015  #?什么意思？
            
        # target_pos[0]+=0.028  # hardcode
        # target_pos[1]+=0.05
            
        print("Predicted target pos: ", target_pos, "Predicted target quat: ", target_quat, "Predicted target gripper: ", target_gripper)
        # if target_gripper==0:
        #     target_gripper=1
        # elif target_gripper==1:
        #     target_gripper=0
        # else:
        #     assert False  # 训练时0开1关，测试时，1开0关
        osc_move(fa,(target_pos, target_quat, np.array([target_gripper])),gripper_thres=gripper_thres,recording=False, recording_continous_frames=False, cameras=None)
    time.sleep(3)

if __name__ == "__main__":
    
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    _eval()