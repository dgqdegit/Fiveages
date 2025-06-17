from frankapy import FrankaArm
from autolab_core import RigidTransform
import numpy as np

import time
from utils.real_camera_utils_new import Camera
import cv2
import os
import json
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

def quat2rotm(quat):
    """Quaternion to rotation matrix.
    quat:w,x,y,z
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    s = w * w + x * x + y * y + z * z
    rotm = np.array(
        [
            [
                1 - 2 * (y * y + z * z) / s,
                2 * (x * y - z * w) / s,
                2 * (x * z + y * w) / s,
            ],
            [
                2 * (x * y + z * w) / s,
                1 - 2 * (x * x + z * z) / s,
                2 * (y * z - x * w) / s,
            ],
            [
                2 * (x * z - y * w) / s,
                2 * (y * z + x * w) / s,
                1 - 2 * (x * x + y * y) / s,
            ],
        ]
    )
    return rotm


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

def gripper_open_flag(fa,gripper_thres):
    gripper_width=fa.get_gripper_width()
    return gripper_width > gripper_thres


def osc_move(fa,target_pose,gripper_thres,recording=False, recording_continous_frames=False, cameras=None):


    target_pos, target_quat, target_gripper = target_pose

    target_pose_fa = RigidTransform(from_frame="franka_tool")
    target_pose_fa.rotation =  quat2rotm(target_quat)  # 注意需要为（w,x,y,z)格式
    target_pose_fa.translation = target_pos


    current_gripper =gripper_open_flag(fa,gripper_thres)# 1 表示开  0 表示关
    last_capture_time = time.time()
    
    fa.goto_pose(target_pose_fa, use_impedance=False) 
    # time.sleep(1)
    if current_gripper != target_gripper[0]:
        
        if target_gripper[0]==1:
            fa.open_gripper()
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)

    pose = fa.get_pose()
    current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
    print("Current pos:", current_pos, "Target pos:", target_pos)
    print("Current Gripper State: ", current_gripper, "Target Gripper State: ", target_gripper[0])

    while np.max(np.abs(current_pos-target_pos)) > 0.02:
        print("The error is too big,retrying!!")
        fa.goto_pose(target_pose_fa, use_impedance=False) 
        pose = fa.get_pose()
        current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
        print("Current pos:", current_pos, "Target pos:", target_pos)     
    while current_gripper != target_gripper[0]:
        if target_gripper[0]==1:
            fa.open_gripper()   
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)
    if recording and not recording_continous_frames:
        result_dict = cameras.capture()
        result_dict['action'] = deepcopy(np.concatenate([target_pos, target_quat, \
                                                        target_gripper]))
        return result_dict





def osc_move_eval(fa,target_pose,gripper_thres,recording=False, recording_continous_frames=False, cameras=None):


    target_pos, target_quat, target_gripper = target_pose

    target_pose_fa = RigidTransform(from_frame="franka_tool")
    target_pose_fa.rotation =  quat2rotm(target_quat)  # 注意需要为（w,x,y,z)格式
    target_pose_fa.translation = target_pos


    current_gripper =gripper_open_flag(fa,gripper_thres)# 1 表示开  0 表示关
    last_capture_time = time.time()
    
    # fa.goto_pose(target_pose_fa, use_impedance=True,duration=6) 
    fa.goto_pose(target_pose_fa, use_impedance=False,duration=5) 
    time.sleep(1)
    if current_gripper != target_gripper[0]:
        
        if target_gripper[0]==1:
            fa.open_gripper()
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)

    pose = fa.get_pose()
    current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
    print("Current pos:", current_pos, "Target pos:", target_pos)
    print("Current Gripper State: ", current_gripper, "Target Gripper State: ", target_gripper[0])

    while np.max(np.abs(current_pos-target_pos)) > 0.02:
        print("The error is too big,retrying!!")
        fa.goto_pose(target_pose_fa, use_impedance=False) 
        pose = fa.get_pose()
        current_pos,current_rot,current_quat = pose.translation, pose.rotation,pose.quaternion
        print("Current pos:", current_pos, "Target pos:", target_pos)     
    while current_gripper != target_gripper[0]:
        if target_gripper[0]==1:
            fa.open_gripper()   
        elif target_gripper[0]==0:
            fa.close_gripper()
        else:
            assert False
        time.sleep(2)
        current_gripper =gripper_open_flag(fa,gripper_thres)
    if recording and not recording_continous_frames:
        result_dict = cameras.capture()
        result_dict['action'] = deepcopy(np.concatenate([target_pos, target_quat, \
                                                        target_gripper]))
        return result_dict




if __name__ == "__main__":

    # # reset_robot
    cameras = Camera(camera_type="all")
    time.sleep(2)
    # task_name = "stack_blocks"    # Change 1
    # instruction = "place the green block on the red block" # Change 2
    # task_idx = 29     # Change 3
    # gripper_open =np.array([True,True,False,False,False,True])  # Change 4  True 开 0 关
    # task_name = "place_block_in_shelf"    # Change 1
    # instruction = "place the red block in top shelf" # Change 2
    # task_idx = 9     # Change 3
    # data_result_dir = "/media/casia/data2/lpy/3D_VLA/realworld_datasets"
    # gripper_open =np.array([True,True,False,False,False,True])  # Change 4  True 开 0 关 

    task_name = "place_block_in_plate"    # Change 1
    instruction = "place the red block in the green plate" # Change 2
    task_idx = 19     # Change 3
    data_result_dir = "/media/casia/data2/lpy/3D_VLA/realworld_datasets"
    gripper_open =np.array([True,True,False,False,False,True])  # Change 4  True 开 0 关 

    save_dir = os.path.join(data_result_dir, task_name,str(task_idx))
    os.makedirs(save_dir, exist_ok=False)
    
    expert_action_file = f"/home/casia/Downloads/task.task"  # override_it_everytime
    action_poses = extract_poses_from_file(expert_action_file)
    
    assert gripper_open.shape[0] == action_poses.shape[0]
    fa = FrankaArm()
    fa.reset_joints()

    dir_keys = ["3rd_cam_imgs", "wrist_cam_imgs", "3rd_cam_rgb", "3rd_cam_depth", "3rd_cam_pcd", \
                "wrist_cam_rgb", "wrist_cam_depth", "wrist_cam_pcd", "actions"]
    
    for key in dir_keys:
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
    

    global_idx = 0 # for recording frames
    for idx, action_pose in enumerate(action_poses):

        action_pose = action_pose.reshape(4,4).T  # 为什么要转置

        action_pose = deepcopy(RigidTransform(
        rotation=action_pose[:3,:3],
        translation=action_pose[:3,3]
        ))
        target_pos, target_quat =  action_pose.translation,action_pose.quaternion # 使用frankapy的transform

        target_gripper = gripper_open[idx:idx+1]  # 
        action = np.concatenate([target_pos, target_quat, target_gripper])
        now_obs=osc_move(fa,(target_pos, target_quat, target_gripper),gripper_thres=0.07,recording=True, recording_continous_frames=False, cameras=cameras)
        # observations.append(deepcopy(now_obs[0]))
        # Save Results
    
        with open(save_dir + f"/actions/{global_idx}.pkl", 'wb') as f:
            pickle.dump(now_obs.pop("action"), f)
            
        for cam_type, cam_values in now_obs.items():
            for img_type, img_values in cam_values.items():
                with open(save_dir + f"/{cam_type}_cam_{img_type}/{global_idx}.pkl", 'wb') as f:
                    pickle.dump(img_values, f)  
            cv2.imwrite(os.path.join(save_dir, f'{cam_type}_cam_imgs', f'{global_idx}.png'), cam_values["rgb"])
        global_idx += 1
        
    with open(save_dir + f"/instruction.pkl", 'wb') as f:
            pickle.dump(instruction, f)
    
    cameras.stop()