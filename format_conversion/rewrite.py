import os

# 要写入的新矩阵内容
new_content = """[[ 0.01358001 -0.62446467 0.78093499 -0.57951051]
 [-0.99958725 -0.0282524 -0.00520942 -0.69267502]
 [ 0.02531639 -0.78054192 -0.62459058 0.59026163]
 [ 0 0 0 1]]"""

# 指定包含子文件夹的主文件夹路径
root_folder = '/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/'  # 修改为你的路径

# 遍历主文件夹中的所有子文件夹
for subdir, dirs, files in os.walk(root_folder):
    for file in files:
        if file == 'extrinsics.txt':
            # 构造完整的文件路径
            file_path = os.path.join(subdir, file)
            print(f"Updating file: {file_path}")  # 打印正在更新的文件路径
            
            # 打开文件并写入新内容
            with open(file_path, 'w') as f:
                f.write(new_content)

print("所有文件已成功更新！")