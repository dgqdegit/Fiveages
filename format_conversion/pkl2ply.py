import pickle  
import numpy as np  
import open3d as o3d  

# 加载点云和RGB图像数据  
def load_data(point_cloud_file, rgb_image_file):  
    with open(point_cloud_file, 'rb') as pc_file:  
        point_cloud = pickle.load(pc_file)  
        
    with open(rgb_image_file, 'rb') as rgb_file:  
        rgb_image = pickle.load(rgb_file)  

    return point_cloud, rgb_image  

# 将点云转换为PLY文件  
def point_cloud_to_ply(point_cloud, rgb_image, ply_file):  
    # 假设point_cloud和rgb_image都是1080 x 1920 x 3的数组  
    points = np.array(point_cloud)  # 1080 x 1920 x 3  
    height, width, _ = rgb_image.shape  

    # 检查点云的维度  
    if points.shape != (height, width, 3):  
        raise ValueError("Point cloud and RGB image dimensions do not match.")  

    # 重塑点云为N x 3形式  
    points_reshaped = points.reshape(-1, 3)  # N x 3  
    rgb_colors = rgb_image.reshape(-1, 3)    # N x 3  

    # 使用Open3D创建PLY文件  
    ply_points = o3d.geometry.PointCloud()  
    ply_points.points = o3d.utility.Vector3dVector(points_reshaped)  
    ply_points.colors = o3d.utility.Vector3dVector(rgb_colors / 255.0)  # 归一化到[0, 1]  

    # 保存为PLY文件  
    o3d.io.write_point_cloud(ply_file, ply_points)  

# 主程序  
if __name__ == "__main__":  
    point_cloud_file = "/home/zk/Desktop/LIb/new_data/point_cloud.pkl"  
    rgb_image_file =  "/home/zk/Desktop/LIb/new_data/rgb_image.pkl"  
    ply_file =  "/home/zk/Desktop/LIb/new_data/outpoint.ply"   

    point_cloud, rgb_image = load_data(point_cloud_file, rgb_image_file)  
    point_cloud_to_ply(point_cloud, rgb_image, ply_file)  

    print("点云已成功转换为PLY文件。")