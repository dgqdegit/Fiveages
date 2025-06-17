import numpy as np  
import matplotlib.pyplot as plt  
import cv2
# 读取npy文件  
point_cloud = np.load('/home/zk/Projects/DobotStudio/vla_data/data/test/data/right_wbl/2025-04-26_18-15-40_r_wbl/zed_depth/0_2025-04-26_18-16-17-431.npy')  # 替换为你的文件名  

# point_cloud.shape = (1080, 1920, 4) , 通道顺序为 X, Y, Z, RGBA  
# 提取Z通道作为深度图  
depth_map = point_cloud[:, :, 2]  

# 打印深度图的形状确认  
print('Depth map shape:', depth_map.shape)  



# 归一化深度图以适合8位图像格式  
# depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)  
# depth_uint8 = depth_normalized.astype(np.uint8)  

# cv2.imwrite('depth_map.png', depth_uint8)


