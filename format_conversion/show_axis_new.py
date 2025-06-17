import open3d as o3d
import numpy as np
from transforms3d.euler import euler2mat
def vis_pcd_with_end(pcd, rgb):  # 点云+原点坐标系+机械臂末端坐标系  
        # 转换点云和颜色形状  
        pcd_flat = pcd.reshape(-1, 3)  
        rgb_flat = rgb.reshape(-1, 3) / 255.0  

        # 创建点云对象  
        pointcloud = o3d.geometry.PointCloud()  
        pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)  
        pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)  

        # 显示原点坐标系  
        axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])  

        # 显示所有内容  
        o3d.visualization.draw_geometries([pointcloud, axis_origin])  

def vis_pcd_with_end(pcd, rgb, end_pose):
    # 转换点云和颜色形状
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # 显示原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])

    # 处理end_pose为浮点数
    end_pose = [float(x) for x in end_pose]
    pos = np.array(end_pose[:3])*0.001
    angles_deg = np.array(end_pose[3:])
    angles_rad = np.deg2rad(angles_deg)  # 角度转弧度

    # 构建旋转矩阵和齐次变换矩阵
    rot_mat = euler2mat(*angles_rad, axes='sxyz')
    # 构造 4x4 齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos

    # 创建末端坐标系并变换
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T)

    # 显示所有内容
    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end])

def vis_pcd_with_end_pred(pcd, rgb, end_pose, pred_pose):
    # 转换点云和颜色形状
    pcd_flat = pcd.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0

    # 创建点云对象
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)

    # 显示原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])

    # -- 处理end_pose
    end_pose = [float(x) for x in end_pose]
    pos_end = np.array(end_pose[:3]) * 0.001
    angles_deg_end = np.array(end_pose[3:])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    axis_end.transform(T_end)

    # -- 处理pred_pose
    pred_pose = [float(x) for x in pred_pose]
    pos_end = np.array(pred_pose[:3]) * 0.001
    angles_deg_end = np.array(pred_pose[3:])
    angles_rad_end = np.deg2rad(angles_deg_end)
    rot_mat_end = euler2mat(*angles_rad_end, axes='sxyz')
    T_end = np.eye(4)
    T_end[:3, :3] = rot_mat_end
    T_end[:3, 3] = pos_end
    target_axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    target_axis_end.transform(T_end)

    # 显示所有内容
    o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end, target_axis_end])
def get_pose(num):
    lines = pose_data.strip().split('\n')  
    third_line = lines[num]  # 提取第2行  current_pose
    value_1, value_2, value_3, value_4, value_5, value_6 = third_line.split()[1:7]   
    pose = [float(value_1), float(value_2), float(value_3),   
                       float(value_4), float(value_5), float(value_6)]  
    return pose

if __name__ == "__main__":
    pcd_pkl_path = "/home/zk/Desktop/LIb/data0/zed_pcd/0.pkl"  
    rgb_pkl_path = "/home/zk/Desktop/LIb/data0/zed_rgb/0.pkl"  
    pose_pkl_path = "/home/zk/Desktop/LIb/data0/pose.pkl"  
    
    import pickle
     
    with open(pose_pkl_path, 'rb') as f:  
        pose_data = pickle.load(f)   
    with open(pcd_pkl_path, 'rb') as f:  
        pcd_data = pickle.load(f)  
    with open(rgb_pkl_path, 'rb') as f:  
        rgb_data = pickle.load(f)   
    
    current_pose = get_pose(1)  #  current_pose
    target_pose = get_pose(2)  # target_pose
 
   
    
    pcd_data = pcd_data[:, :, :3]
    pcd_data = pcd_data.reshape(-1, 3)
    rgb_data = rgb_data[:, :, :3]
    rgb_data = rgb_data.reshape(-1, 3)
    vis_pcd_with_end_pred(pcd_data,rgb_data,current_pose,target_pose)
    