from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys import config_root

import time
from utils.real_camera_utils import Camera
import cv2
import os
import json
import numpy as np
import pickle


logger = get_deoxys_example_logger()


def extract_poses_from_file(file_path):
    """
    Reads the JSON content from the specified file and extracts
    all 'pose' arrays from the 'parameterized_poses' field.
    """
    # Read the file and parse the JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Navigate to "parameterized_poses"
    parameterized_poses = data.get("parameter", {}).get("parameterized_poses", [])
    
    # Collect the "pose" array from each item if present
    poses = []
    for item in parameterized_poses:
        pose_data = item.get("pose_with_joint_angles", {}).get("pose", [])
        poses.append(pose_data)
    
    return np.array(poses)


def osc_move(robot_interface, controller_type, controller_cfg, target_pose, num_steps, \
                    recording=False, recoeding_continous_frames=False, cameras=None):
    
    observations = []
    
    target_pos, target_quat, target_gripper = target_pose
    current_rot, current_pos = robot_interface.last_eef_rot_and_pos
    current_gripper = robot_interface.last_gripper_action
    last_capture_time = time.time()
    
    # If gripper state changes, act twice
    # print("Current Gripper State: ", current_gripper, "Target Gripper State: ", target_gripper[0])
    if current_gripper != target_gripper[0] and current_gripper != 0:
        for _ in range(10):
            current_pose = robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() *1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -1, 1)

            # First do not change gripper state
            action = action_pos.tolist() + action_axis_angle.tolist() + [current_gripper]
            # print(np.round(action, 2))
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        
            if recording and recoeding_continous_frames and (time.time()-last_capture_time) > 1 and np.linalg.norm(action_pos) / 10.0 > 0.02:
                result_dict = cameras.capture()
                result_dict['action'] = np.concatenate([current_pos, current_quat, \
                                                            [robot_interface.last_gripper_action]])
                observations.append(result_dict)
                last_capture_time = time.time()
        
        
        for _ in range(num_steps-10):
            current_pose = robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() *1
            action_axis_angle = axis_angle_diff.flatten() *1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -1, 1)

            # First do not change gripper state
            action = action_pos.tolist() + action_axis_angle.tolist() + [current_gripper]
            # print(np.round(action, 2))
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        
            if recording and recoeding_continous_frames and (time.time()-last_capture_time) > 1 and np.linalg.norm(action_pos) / 10.0 > 0.02:
                result_dict = cameras.capture()
                result_dict['action'] = np.concatenate([current_pos, current_quat, \
                                                            [robot_interface.last_gripper_action]])
                observations.append(result_dict)
                last_capture_time = time.time()
        
        num_steps = 70
            
    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = (target_pos - current_pos).flatten() * 10
        action_axis_angle = axis_angle_diff.flatten() * 1
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)


        action = action_pos.tolist() + action_axis_angle.tolist() + target_gripper.tolist()
        # print(np.round(action, 2))
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    
            
        if recording and recoeding_continous_frames and (time.time()-last_capture_time) > 1 and np.linalg.norm(action_pos) / 10.0 > 0.02:
            result_dict = cameras.capture()
            result_dict['action'] = np.concatenate([current_pos, current_quat, \
                                                        [robot_interface.last_gripper_action]])
            observations.append(result_dict)
            last_capture_time = time.time()
    
    if num_steps == 70:
        time.sleep(0.5)
        
    print("Current pos:", current_pos, "Target pos:", target_pos)
            
    if recording and not recoeding_continous_frames:
        result_dict = cameras.capture()
        result_dict['action'] = np.concatenate([target_pos, target_quat, \
                                                        [robot_interface.last_gripper_action]])
        observations.append(result_dict)
        
    return observations


if __name__ == "__main__":

    # # reset_robot
    # with open("deoxys_control/deoxys/examples/reset_robot_joints.py", "r") as file:
    #     code = file.read()
    #     exec(code)
    
    cameras = Camera(camera_type="all")
    time.sleep(2)

    task_name = "put_in_drawer"    # Change 1
    task_idx = 11    # Change 2
    instruction = "put the blue block in the drawer" # Change 3
    data_result_dir = "/data2/cyx/realworld_datasets"
    
    save_dir = os.path.join(data_result_dir, task_name,str (task_idx))
    os.makedirs(save_dir, exist_ok=False)
    
    # expert_action_file = f"/home/lenovo/Downloads/{task_name}_{task_idx}.task"
    expert_action_file = f"/home/lenovo/Downloads/task.task"  
    action_poses = extract_poses_from_file(expert_action_file)
    
    # gripper_open = [True, True, False, False, False, True]    # Change 4
    # gripper_open = [True, False, False] 
    gripper_open = [True, True, False, True, True, False, False, True] 
    # gripper_open = np.random.choice([True, False], size=action_poses.shape[0])
    gripper_open = np.array([-0.9 if i else 0.9 for i in gripper_open])
    
    assert gripper_open.shape[0] == action_poses.shape[0]
    
    interface_cfg = "charmander.yml"
    controller_type = "OSC_POSE"
    
    robot_interface = FrankaInterface(
        config_root + f"/{interface_cfg}", use_visualizer=False
    )
    controller_cfg = get_default_controller_config(controller_type)
    
    dir_keys = ["3rd_cam_imgs", "wrist_cam_imgs", "3rd_cam_rgb", "3rd_cam_depth", "3rd_cam_pcd", \
                "wrist_cam_rgb", "wrist_cam_depth", "wrist_cam_pcd", "actions"]
    
    for key in dir_keys:
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
    
    observations = []

    global_idx = 0 # for recording frames
    for idx, action_pose in enumerate(action_poses):
        print(robot_interface.last_gripper_action)
        action_pose = action_pose.reshape(4,4).T
        
        target_pos, target_quat = transform_utils.mat2pose(action_pose)
        target_gripper = gripper_open[idx:idx+1]
        
        action = np.concatenate([target_pos, target_quat, target_gripper])
    
        observations += osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat, target_gripper),
            num_steps=120,
            recording=True,
            recoeding_continous_frames=False,
            cameras=cameras,
        )

    # Save Results
    for cam_observations in observations:
        with open(save_dir + f"/actions/{global_idx}.pkl", 'wb') as f:
            pickle.dump(cam_observations.pop("action"), f)
            
        for cam_type, cam_values in cam_observations.items():
            for img_type, img_values in cam_values.items():
                with open(save_dir + f"/{cam_type}_cam_{img_type}/{global_idx}.pkl", 'wb') as f:
                    pickle.dump(img_values, f)  
            cv2.imwrite(os.path.join(save_dir, f'{cam_type}_cam_imgs', f'{global_idx}.png'), cam_values["rgb"])
        global_idx += 1
        
    with open(save_dir + f"/instruction.pkl", 'wb') as f:
            pickle.dump(instruction, f)
    
    cameras.stop()