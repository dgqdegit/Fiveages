import open3d as o3d  
import numpy as np  
import pickle  
from transforms3d.euler import euler2mat  

def get_cam_extrinsic(type):  
    """  
    获取相机的外部参数矩阵  
    """  
    if type == "3rd":  
        transform = [[ 0.03910006,  0.63361241, -0.77266195,  0.57975187],  
                     [ 0.99908312, -0.01129462,  0.04129593, -0.53077248],  
                     [ 0.01743869, -0.77356819, -0.63347309,  0.57926681],  
                     [ 0.,          0.,          0.,          1.        ]]  
    elif type == "wrist":  
        transform = np.array([[0.6871684912377796, -0.7279882263970943, 0.8123566411202088, 0.0],  
                              [-0.869967706085017, -0.2561670369853595, 0.13940123346877276, 0.0],  
                              [0.0, 0.0, 0.0, 1.0]])  
    else:  
        raise ValueError("Invalid type")  
    
    return np.array(transform)    

class MyVisualizer:  
    def vis_pcd_with_end(self, pcd, rgb, end_pose):  
        # 转换点云和颜色形状  
        pcd_flat = pcd.reshape(-1, 3)  
        rgb_flat = rgb.reshape(-1, 3) / 255.0  

        # 创建点云对象  
        pointcloud = o3d.geometry.PointCloud()  
        pointcloud.points = o3d.utility.Vector3dVector(pcd_flat)  
        pointcloud.colors = o3d.utility.Vector3dVector(rgb_flat)  

        # 显示原点坐标系  
        axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])  

        # 处理 end_pose 为浮点数  
        end_pose = [float(x) for x in end_pose]  
        pos = np.array(end_pose[:3]) * 0.001  
        angles_deg = np.array(end_pose[3:])  
        angles_rad = np.deg2rad(angles_deg)  # 角度转弧度  

        # 构建旋转矩阵和齐次变换矩阵  
        rot_mat = euler2mat(*angles_rad, axes='sxyz')  
        T = np.eye(4)  
        T[:3, :3] = rot_mat  
        T[:3, 3] = pos  

        # 创建末端坐标系并变换  
        axis_end = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  
        axis_end.transform(T)  

        # 显示所有内容  
        o3d.visualization.draw_geometries([pointcloud, axis_origin, axis_end])  

if __name__ == "__main__":  
    # 数据路径配置  
     
    pcd_pkl_path = "/home/zk/Desktop/LIb/new_data/point_cloud.pkl"  
    rgb_pkl_path ="/home/zk/Desktop/LIb/new_data/rgb_image.pkl"  
    pose_pkl_path = "/home/zk/Desktop/LIb/data0/pose.pkl"  
    
    # 加载末端位姿数据  
    # with open(pose_pkl_path, 'rb') as f:  
    #     pose_data = pickle.load(f)   
      
    # 加载点云数据  
    with open(pcd_pkl_path, 'rb') as f:  
        pcd_data = pickle.load(f)   
    pcd_data = pcd_data[:, :, :3]  # 只保留到 RGB 通道  
    pcd_data = pcd_data.reshape(-1, 3)  # 转换为 (N, 3)  

    # 加载 RGB 数据  
    with open(rgb_pkl_path, 'rb') as f:  
        rgb_data = pickle.load(f)  
    rgb_data = rgb_data[:, :, :3]  # 只保留到 RGB 通道  
    rgb_data = rgb_data.reshape(-1, 3)  # 转换为 (N, 3)  

    # 提取 end_pose 值  
    pose_data = [48.5491,-200,148.3259,90,-5,-90]
    # lines = pose_data.strip().split('\n')  
    # third_line = lines[2]  # 提取第三行  
    # print(third_line)
    
    value_1, value_2, value_3, value_4, value_5, value_6 = pose_data
    
    # 转换为浮点数  
    end_pose_values = [float(value_1), float(value_2), float(value_3),   
                       float(value_4), float(value_5), float(value_6)]  
    # print(end_pose_values)
    # exit()
    # 转换点云到机器人基座坐标系  
    cam_type = "3rd"  # 假设您需要转换“3rd”相机  
    transform_matrix = get_cam_extrinsic(cam_type)  
    
    # 添加一列 1 以转换为齐次坐标  
    pcd_data_homogeneous = np.concatenate((pcd_data, np.ones((pcd_data.shape[0], 1))), axis=1)  
    
    # 转换到基座坐标系  
    pcd_base = (transform_matrix @ pcd_data_homogeneous.T).T[:, :3]  
    
    # 创建可视化器实例并显示结果  
    visualizer = MyVisualizer()  
    visualizer.vis_pcd_with_end(pcd_base, rgb_data, end_pose_values)