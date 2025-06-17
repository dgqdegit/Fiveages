import os
import shutil
import pickle
import numpy as np
import transforms3d

def convert_rgb(franka_data_root, dobot_data_root, data_name):
    dobot_data_dir = os.path.join(dobot_data_root, data_name, "zed_rgb")
    franka_data_dir = os.path.join(franka_data_root, data_name, "3rd_cam_rgb")
    # copy franka_data_dir to dobot_data_dir
    os.makedirs(dobot_data_dir, exist_ok=True)
    for filename in os.listdir(franka_data_dir):
        src_file = os.path.join(franka_data_dir, filename)
        dst_file = os.path.join(dobot_data_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

def convert_pcd(franka_data_root, dobot_data_root, data_name):
    dobot_data_dir = os.path.join(dobot_data_root, data_name, "zed_pcd")
    franka_data_dir = os.path.join(franka_data_root, data_name, "3rd_cam_pcd")
    # copy franka_data_dir to dobot_data_dir
    os.makedirs(dobot_data_dir, exist_ok=True)
    for filename in os.listdir(franka_data_dir):
        src_file = os.path.join(franka_data_dir, filename)
        dst_file = os.path.join(dobot_data_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

def convert_pose(franka_data_root, dobot_data_root, data_name):
    franka_data_dir = os.path.join(franka_data_root, data_name, "actions")
    dobot_data_path = os.path.join(dobot_data_root, data_name, "pose.pkl")
    first_line = "Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz)"
    pose_list = os.listdir(franka_data_dir)
    pose_lines = [first_line]

    for i in range (len(pose_list)):
        filename = f"{i}.pkl"
        src_file = os.path.join(franka_data_dir, filename)
        if not os.path.exists(src_file):
            continue
        with open(src_file, "rb") as f:
            pose = pickle.load(f)
            t = pose[0:3]*1000.
            r_quat = pose[3:7]
            griper_state = pose[-1]
            r_euler = transforms3d.euler.quat2euler(r_quat, axes='sxyz')
            r_euler = np.rad2deg(r_euler)
            combined = np.concatenate([t, r_euler])  # 合并成一个一维数组
            result_str = ' '.join(str(x) for x in combined)
            result_str = '202505300000 ' + result_str + f' {int(griper_state)}'
            pose_lines.append(result_str)

    # 创建目标目录（如果还没有）
    os.makedirs(os.path.dirname(dobot_data_path), exist_ok=True)
    # 一次性保存所有pose
    with open(dobot_data_path, "wb") as f:
        p_lines = '\n'.join(x for x in pose_lines)
        pickle.dump(p_lines, f)


def convert_extrinsic(franka_data_root, dobot_data_root, data_name):
    franka_extrinsic_path = os.path.join(franka_data_root, data_name, "extrinsic_matrix.npy")
    dobot_extrinsic_path = os.path.join(dobot_data_root, data_name, "extrinsic_matrix.pkl")
    # 读取 extrinsic_matrix.npy
    extrinsic_matrix = np.load(franka_extrinsic_path)

    # 创建目标目录（如果不存在）
    os.makedirs(os.path.dirname(dobot_extrinsic_path), exist_ok=True)

    # 保存为 pickle 文件
    with open(dobot_extrinsic_path, "wb") as f:
        pickle.dump(extrinsic_matrix, f)

def convert_instructions(franka_data_root, dobot_data_root, data_name):
    franka_extrinsic_path = os.path.join(franka_data_root, data_name, "instruction.pkl")
    dobot_extrinsic_path = os.path.join(dobot_data_root, data_name, "instruction.pkl")
    shutil.copy(franka_extrinsic_path, dobot_extrinsic_path)



def convert_franka_2_dobot(franka_data_root, dobot_data_root):
    data_list = os.listdir(franka_data_root)
    for data_name in data_list:
        print("Converting {}...".format(data_name))
        convert_rgb(franka_data_root, dobot_data_root, data_name)
        convert_pcd(franka_data_root, dobot_data_root, data_name)
        convert_pose(franka_data_root, dobot_data_root, data_name)
        convert_extrinsic(franka_data_root, dobot_data_root, data_name)
        convert_instructions(franka_data_root, dobot_data_root, data_name)


if __name__ == '__main__':
    franka_data_root = "/home/zk/Projects/Datasets/put_shelf_0611_right"
    dobot_data_root = "/home/zk/Projects/Datasets/dobot_formate_put_shelf_0611_right"
    convert_franka_2_dobot(franka_data_root, dobot_data_root)