import open3d as o3d
import os
import pickle
import numpy as np
from transforms3d.euler import euler2mat

def get_pose(pose_data, num):
    lines = pose_data.strip().split('\n')
    third_line = lines[num]  # Extract line corresponding to current pose
    value_1, value_2, value_3, value_4, value_5, value_6 = third_line.split()[1:7]
    pose = [float(value_1), float(value_2), float(value_3), float(value_4), float(value_5), float(value_6)]
    return pose


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
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
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
    rot_mat_pred = euler2mat(*angles_rad_pred, axes='sxyz')
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
    data_path = "/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/0"
    # data_path = "/home/zk/Projects/DobotStudio/vla_data/data/data/right_wbl/0"
    # data_path = "/home/zk/Public/datasets/put_bottle_in_microwave2/26"  # （偶数测试集，奇数训练集）
    pcd_dir = os.path.join(data_path, "zed_pcd")
    rgb_dir = os.path.join(data_path, "zed_rgb")
    pose_path = os.path.join(data_path, "pose.pkl")
    extrinsic_matrix = os.path.join(data_path, "extrinsic_matrix.pkl")

    with open(pose_path, 'rb') as f:
        pose_data = pickle.load(f)
    print("---------")
    print(type(pose_data))

    with open(extrinsic_matrix, 'rb') as f:
        extrinsic_matrix = pickle.load(f)
        extrinsic_matrix = np.array(extrinsic_matrix)
    print(extrinsic_matrix)
    
    print(type(extrinsic_matrix))
    # exit()
    # 显示第i组数据
    for i in range(4):
        pcd_path = os.path.join(pcd_dir, f"{i}.pkl")
        rgb_path = os.path.join(rgb_dir, f"{i}.pkl")
        pose_current = get_pose(pose_data, i+1)
        pose_next = get_pose(pose_data, i+2)
        with open(pcd_path, 'rb') as f:
            pcd_data = pickle.load(f)
        with open(rgb_path, 'rb') as f:
            rgb_data = pickle.load(f)

        pcd_data = pcd_data[:, :, :3]
        rgb_data = rgb_data[:, :, :3]
        vis_pcd_with_end_pred(pcd_data, rgb_data, extrinsic_matrix, pose_current, pose_next)


