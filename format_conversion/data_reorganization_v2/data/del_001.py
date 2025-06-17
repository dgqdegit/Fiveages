import os

def remove_first_line_from_txt_files(parent_directory):
    """ 遍历每个子文件夹，去掉指定 TXT 文件的第一行 """
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(parent_directory, folder)  # 子文件夹完整路径
        
        # 针对每个需要处理的 TXT 文件名
        txt_files = ['extrinsics.txt']
        
        for txt_file in txt_files:
            txt_file_path = os.path.join(folder_path, txt_file)
            if os.path.isfile(txt_file_path):
                # 读取文件内容并去掉第一行
                with open(txt_file_path, 'r') as file:
                    lines = file.readlines()
                
                # 只保留第一行之后的内容
                new_lines = lines[1:]  # 去掉第一行

                # 保存处理后的内容
                with open(txt_file_path, 'w') as file:
                    file.writelines(new_lines)

                print(f"Removed first line from: {txt_file_path}")

if __name__ == "__main__":
    # 设置主目录路径
    parent_directory = '/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/'   # 请根据需要修改路径
    remove_first_line_from_txt_files(parent_directory)