import os
import numpy as np
import pickle

# 指定要处理的文件夹路径
folder_path = '/home/zk/Projects/Datasets/right_wbl_0529'  # 替换为实际的文件夹路径

# 遍历文件夹中的所有子文件夹
for subdir, _, files in os.walk(folder_path):
    for file in files:
        # 只处理npy文件
        if file.endswith('.npy'):
            npy_file_path = os.path.join(subdir, file)
            
            # 读取npy文件内容
            data = np.load(npy_file_path, allow_pickle=True)
            
            # 保存为pkl文件
            pkl_file_path = npy_file_path.replace('.npy', '.pkl')
            with open(pkl_file_path, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)
            
            # 删除原始npy文件
            os.remove(npy_file_path)

print("转换完成！")
