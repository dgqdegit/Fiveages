import numpy as np  
import open3d as o3d  

def load_and_visualize_zed_pointcloud(file_path):  
    """  
    加载保存的ZED点云.npy文件（XYZRGBA格式）并可视化。  
    """  
    print(f"Loading point cloud data from: {file_path}")  
    pointcloud_rgba = np.load(file_path)  # shape是(H, W, 4)，4通道为 XYZ + RGB  

    # 确认点云数据形状  
    H, W, C = pointcloud_rgba.shape  
    print(f"Original point cloud shape: {pointcloud_rgba.shape}")  

    # 重塑为N x 4 (每个点X,Y,Z和颜色)  
    points_rgba = pointcloud_rgba.reshape(-1, 4)  

    # 过滤无效点：去除坐标异常或0的点  
    valid_mask = (  
        np.isfinite(points_rgba[:, 0]) &  
        np.isfinite(points_rgba[:, 1]) &  
        np.isfinite(points_rgba[:, 2]) &  
        (points_rgba[:, 2] != 0)  
    )  
    valid_points = points_rgba[valid_mask]  

    print(f"Valid points count: {valid_points.shape[0]}")  

    # 构造Open3D点云对象  
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  

    

    # 显示点云  
    o3d.visualization.draw_geometries([pcd], window_name="ZED Point Cloud Visualization")  

    return pcd  # 返回点云对象，以便后续保存  

def save_point_cloud_to_ply(pcd, save_path):  
    """  
    将点云保存为.ply文件  
    """  
    print(f"Saving point cloud to: {save_path}")  
    o3d.io.write_point_cloud(save_path, pcd)  
    print("Point cloud saved successfully.")  

if __name__ == "__main__":  
    # 修改为你的npy点云文件路径  
    npy_path = "/home/zk/Projects/DobotStudio/vla_data/data/test/data/right_wbl/2025-04-26_18-15-40_r_wbl/zed_depth/0_2025-04-26_18-16-17-431.npy"  

    # 加载和可视化点云  
    pcd = load_and_visualize_zed_pointcloud(npy_path)  

    # 保存点云为.ply文件  
    # ply_save_path = "/home/zk/Projects/DobotStudio//output_point_cloud.ply"  
    # save_point_cloud_to_ply(pcd, ply_save_path)