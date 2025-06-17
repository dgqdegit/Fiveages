import pickle  
import numpy as np  
import open3d as o3d  

# 从pkl文件加载点云数据  
pkl_file_path = '/home/zk/Projects/DobotStudio/vla_data/data/test/point_cloud.pkl'  
with open(pkl_file_path, 'rb') as f:  
    points = pickle.load(f)  

# 将点云数据转换为numpy数组  
points = np.array(points)  

# 创建Open3D点云对象  
pcd = o3d.geometry.PointCloud()  
pcd.points = o3d.utility.Vector3dVector(points)  

# 可视化点云  
o3d.visualization.draw_geometries([pcd])