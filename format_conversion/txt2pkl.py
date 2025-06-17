import os
import pandas as pd
import pickle

# 指定要处理的文件夹路径
folder_path = '/home/zk/Projects/Datasets/open_the_door_0610'  # 替换为实际的文件夹路径

# 遍历文件夹中的所有子文件夹
for subdir, _, files in os.walk(folder_path):
    for file in files:
        # 只处理txt文件
        if file.endswith('.txt'):
            txt_file_path = os.path.join(subdir, file)
            
            # 读取txt文件内容
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                data = f.read()
            
           
            
            # 保存为pkl文件
            pkl_file_path = txt_file_path.replace('.txt', '.pkl')
            with open(pkl_file_path, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)
            
            # 删除原始txt文件
            os.remove(txt_file_path)

print("转换完成！")
