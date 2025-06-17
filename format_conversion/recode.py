import os
import numpy as np
import pickle
import ast

# 定义文件路径
base_directory = '/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1'  # 主目录路径

def convert_images_to_pkl(image_folder, pkl_folder):
    """ 将jpg图片转换为pkl文件并移动到指定文件夹 """
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.jpg'):
            # 构建完整文件路径
            img_path = os.path.join(image_folder, img_file)
            # 读取图片并转换为numpy数组 (示例中假设使用PIL读取图片)
            from PIL import Image
            
            # 读取图像
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 转换为pkl文件路径
            pkl_file = os.path.join(pkl_folder, img_file.replace('.jpg', '.pkl'))
            
            # 保存为pkl文件
            with open(pkl_file, 'wb') as f:
                pickle.dump(img_array, f)
    
    print(f"Converted images in {image_folder} to PKL format and saved to {pkl_folder}.")

def format_matrix_data(matrix_data):
    import re
    # 用正则表达式找到不规范的行，并加入逗号
    return re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', matrix_data)

def process_txt_files(folder_path):
    """ 处理三个TXT文件并转换为PKL格式 """
    # 处理pose.txt
    pose_file_path = os.path.join(folder_path, 'pose.txt')
    with open(pose_file_path, 'r') as pose_file:
        lines = pose_file.readlines()
    
    # 修改pose.txt文件内容
    if len(lines) >= 5:
        for i in range(2, 5):  # 修改第三到第五行
            lines[i] = lines[i].rstrip() + ' 1\n'
    for i in range(len(lines)):
        if i < 2 or i > 4:
            lines[i] = lines[i].rstrip() + '\n'
    
    # 保存修改后的内容
    with open(pose_file_path, 'w') as pose_file:
        pose_file.writelines(lines)

    pose_file_path = os.path.join(folder_path, 'pose.txt')
    # 将修改后的pose.txt转换为pkl文件
    with open(pose_file_path, 'r') as instr_file:
        pose_data = instr_file.read()
    
    with open(pose_file_path.replace('.txt', '.pkl'), 'wb') as f:
        pickle.dump(pose_data, f)

    # 处理extrinsics.txt，重命名并转换
    extrinsics_file_path = os.path.join(folder_path, 'extrinsics.txt')
    extrinsic_matrix_file_path = os.path.join(folder_path, 'extrinsic_matrix.txt')
    os.rename(extrinsics_file_path, extrinsic_matrix_file_path)  # 重命名文件

    with open(extrinsic_matrix_file_path, 'r') as file:
    # 读取内容并去掉前后空白字符
        string = file.read().strip()

    # 去掉方括号
    string = string.replace('[', '').replace(']', '').strip()

    # 用空白分隔行，假设行与行之间存在换行符
    rows = string.splitlines()

    # 将每一行转换为numpy数组
    matrix_numpy = np.array([[float(num) for num in row.split()] for row in rows])

    # 现在 matrix_numpy 应该是 4x4 矩阵
    print(matrix_numpy.shape)  # 应该输出 (4, 4)

    # 将numpy数组保存为pkl文件
    with open(extrinsic_matrix_file_path.replace('.txt', '.pkl'), 'wb') as f:
        pickle.dump(matrix_numpy, f)
    # 处理instruction.txt
    instruction_file_path = os.path.join(folder_path, 'instruction.txt')
    # 直接转换为pkl文件
    with open(instruction_file_path, 'r') as instr_file:
        instruction_data = instr_file.read()
    
    with open(instruction_file_path.replace('.txt', '.pkl'), 'wb') as f:
        pickle.dump(instruction_data, f)

    print(f"Processed TXT files in {folder_path} and converted to PKL format.")

# 主程序
def main():
    # 获取子文件夹
    subfolders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]
    
    # 根据序号重命名子文件夹
    for index, folder in enumerate(subfolders):
        new_folder_name = str(index)
        old_folder_path = os.path.join(base_directory, folder)
        new_folder_path = os.path.join(base_directory, new_folder_name)
        os.rename(old_folder_path, new_folder_path)  # 重命名

        # 进入新的文件夹进行处理
        realsense_imgs_folder = os.path.join(new_folder_path, 'realsense_imgs')
        realsense_rgb_folder = os.path.join(new_folder_path, 'realsense_rgb')
        zed_imgs_folder = os.path.join(new_folder_path, 'zed_imgs')
        zed_rgb_folder = os.path.join(new_folder_path, 'zed_rgb')
        zed_pcd_folder = os.path.join(new_folder_path, 'zed_pcd')
        zed_depth_folder = os.path.join(new_folder_path, 'zed_depth')

        # 创建目标文件夹，如果不存在
        os.makedirs(realsense_rgb_folder, exist_ok=True)
        os.makedirs(zed_rgb_folder, exist_ok=True)

        # 执行图像转换
        convert_images_to_pkl(realsense_imgs_folder, realsense_rgb_folder)
        convert_images_to_pkl(zed_imgs_folder, zed_rgb_folder)

        # 处理深度和点云图像 (转换npz文件为pkl)
        for depth_file in os.listdir(zed_depth_folder):
            if depth_file.endswith('.npy'):
                depth_file_path = os.path.join(zed_depth_folder, depth_file)
                depth_array = np.load(depth_file_path)
                with open(depth_file_path.replace('.npy', '.pkl'), 'wb') as f:
                    pickle.dump(depth_array, f)

        for pcd_file in os.listdir(zed_pcd_folder):
            if pcd_file.endswith('.npy'):
                pcd_file_path = os.path.join(zed_pcd_folder, pcd_file)
                pcd_array = np.load(pcd_file_path)
                with open(pcd_file_path.replace('.npy', '.pkl'), 'wb') as f:
                    pickle.dump(pcd_array, f)

        # 处理TXT文件
        process_txt_files(new_folder_path)

if __name__ == "__main__":
    main()