import open3d as o3d
import os
import pickle
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
import transforms3d
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler_xyz(quaternion):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) using XYZ convention.

    Parameters:
        quaternion: A tuple of quaternion (w, x, y, z)

    Returns:
        A tuple of Euler angles (roll, pitch, yaw) in radians
    """
    w, x, y, z = quaternion

    # Calculate roll (rotation around X-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (rotation around Y-axis)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Calculate yaw (rotation around Z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
def get_pose(pose_dir, num):
    pose_path = os.path.join(pose_dir, f"{num}.pkl")
    with open(pose_path, "rb") as f:
        pose_data = pickle.load(f)
    pose_quat = pose_data[3:7]
    gripper_state = pose_data[-1]
    # pose_eurl = quaternion_to_euler_xyz(pose_quat)
    pose_mat = transforms3d.quaternions.quat2mat(pose_quat)
    # pose_eurl = mat2euler(pose_mat, axes='sxyz')
    # pose_quat_ = [pose_quat[-1], pose_quat[0], pose_quat[1], pose_quat[2]]
    pose_eurl = transforms3d.euler.quat2euler(pose_quat)
    pose_eurl_1 = np.rad2deg(np.asarray(pose_eurl))

    pose_quat_ = [pose_quat[-1], pose_quat[0], pose_quat[1], pose_quat[2]]
    rotation = R.from_quat(pose_quat_)
    pose_eurl = rotation.as_euler('xyz')
    pose_eurl_2 = np.rad2deg(np.asarray(pose_eurl))

    print("quat:", pose_quat)
    print("eurl:", pose_eurl)
    pose = [float(pose_data[0]*1000.), float(pose_data[1]*1000.), float(pose_data[2]*1000.), float(pose_eurl[0]), float(pose_eurl[1]), float(pose_eurl[2])]
    return pose, gripper_state


def convert_pcd_to_base(extrinsic_matrix, pcd=[]):
    transform = extrinsic_matrix

    h, w = pcd.shape[:2]
    pcd = pcd.reshape(-1, 3)

    pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd = (transform @ pcd.T).T[:, :3]

    pcd = pcd.reshape(h, w, 3)
    return pcd


def vis_pcd_with_end_pred(pcd, rgb, extrinsic_matrix, end_pose, pred_pose):
    # Convert point cloud coordinates
    pcd = convert_pcd_to_base(extrinsic_matrix, pcd)

    # Convert point cloud and colors to flat shapes
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # Create point cloud object
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # Display origin coordinate frame
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # Process end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3]) * 0.001
    angles_deg_end = np.array(end_pose[3:])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='rxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    axis_end.transform(T_end)

    # Process pred_pose
    pred_pose = [float(x) for x in pred_pose]
    pos_pred = np.array(pred_pose[:3]) * 0.001
    angles_deg_pred = np.array(pred_pose[3:])
    angles_rad_pred = np.deg2rad(angles_deg_pred)
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='rxyz')
    T_pred = np.eye(4)
    T_pred[:3, :3] = rot_mat_pred
    T_pred[:3, 3] = pos_pred
    target_axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    target_axis_end.transform(T_pred)

    print("111")
    # Show all content
    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end, target_axis_end])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data_path = "/home/zk/Projects/DobotStudio/vla_data/data/right_wbl_1/1"
    # data_path = "/home/zk/Projects/DobotStudio/vla_data/data/data/right_wbl/0"
    # data_path = "/home/zk/Projects/DobotStudio/vla_data/data/right_wbl_0528_1/data/0"  # （偶数测试集，奇数训练集）
    data_path = "/home/zk/Projects/Datasets/put_cube_0611_right/6"   # 数据集路径
    pcd_dir = os.path.join(data_path, "3rd_cam_pcd")
    rgb_dir = os.path.join(data_path, "3rd_cam_rgb")
    pose_dir = os.path.join(data_path, "actions")
    extrinsic_matrix = os.path.join(data_path, "extrinsic_matrix.npy")

    # with open(pose_path, 'rb') as f:
    #     pose_data = pickle.load(f)

    with open(extrinsic_matrix, 'rb') as f:
        extrinsic_matrix = np.load(f)
        # extrinsic_matrix = np.array(extrinsic_matrix)
    print(extrinsic_matrix)
    # print(type(extrinsic_matrix))
    # exit()
    # 显示第i组数据
    step = 5
    for i in range(5):
        pcd_path = os.path.join(pcd_dir, f"{i}.pkl")
        rgb_path = os.path.join(rgb_dir, f"{i}.pkl")

        pose_current, gripper_current = get_pose(pose_dir, i)
        if i < step - 2:
            pose_next, gripper_next = get_pose(pose_dir, i + 1)
        else:
            pose_next, gripper_next = pose_current, gripper_current
        with open(pcd_path, 'rb') as f:
            pcd_data = pickle.load(f)
        with open(rgb_path, 'rb') as f:
            rgb_data = pickle.load(f)

        
        pcd_data = pcd_data[:, :, :3]
        rgb_data = rgb_data[:, :, :3]
        vis_pcd_with_end_pred(pcd_data, rgb_data, extrinsic_matrix, pose_current, pose_next)



