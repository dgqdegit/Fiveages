import os

# 要写入的字符串
data = """[[0.01358001 -0.62446467 0.78093499 -0.57951051]
[-0.99958725 -0.0282524 -0.00520942 -0.69267502]
[0.02531639 -0.78054192 -0.62459058 0.59026163]
[0 0 0 1]]"""

# 根目录路径，替换为你的文件夹路径
root_directory = '/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/'  # 这里替换成你的目录路径

# 遍历每个子文件夹
for subdir, _, files in os.walk(root_directory):
    if 'extrinsics.txt' in files:
        file_path = os.path.join(subdir, 'extrinsics.txt')
        
        # 写入数据到 extrinsic_matrix.txt
        with open(file_path, 'w') as file:
            file.write(data)
        
        print(f"数据已写入: {file_path}")