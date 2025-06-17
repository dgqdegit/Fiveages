import os
import numpy as np
import pickle

# 指定要处理的文件夹路径
folder_path = '/home/zk/Projects/Datasets/right_wbl_0529'  # 替换为实际的文件夹路径

for subdir, _, files in os.walk(folder_path):
    for file in files:
        # 只处理pkl文件
        if file.endswith('.pkl'):
            pkl_file_path = os.path.join(subdir, file)
            
            # 读取pkl文件内容
            with open(pkl_file_path, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            
            # 保存为npy文件
            npy_file_path = pkl_file_path.replace('.pkl', '.npy')
            np.save(npy_file_path, data)
            
            # 删除原始pkl文件
            os.remove(pkl_file_path)

print("转换完成！")
