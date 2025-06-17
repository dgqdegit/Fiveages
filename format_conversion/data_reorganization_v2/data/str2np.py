
txt_file_path = "/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/11/pose.txt"  # 替换为你的实际文件路径


import numpy as np
import ast

# # 读取 txt 文件
with open(txt_file_path, 'r') as file:
    # 读取文件内容
    content = file.read()

# # 将文件内容中的空格替换为逗号，并确保格式正确
# # 处理行与行之间的分隔
# print(content)
# print(type(content))

# formatted_content = content.replace('] [', '],[').replace(' ', ',').strip()

# 使用 ast.literal_eval() 将格式化字符串转换为列表
# list_matrix = ast.literal_eval(formatted_content)

# 转换为 numpy.array
# array_matrix = np.array(list_matrix)

# print(array_matrix)
# print(type(array_matrix))


print(content)
print(type(content))