import numpy as np

# 从 txt 文件加载字符串
file_path = '/home/zk/Projects/DobotStudio/vla_data/data/test/data/left_wbl_1/25/extrinsic_matrix.txt'  # 假设文件名为 data.txt
with open(file_path, 'r') as file:
    # 读取内容并去掉前后空白字符
    string = file.read().strip()

# 去掉方括号，然后把字符串转换为 numpy 数组
# 先用空格替换内部的换行符
string = string.replace('[', '').replace(']', '').replace('\n', ' ').strip()

# 用空格分割每一行再转换成 numpy 数组
array = np.array([[float(num) for num in row.split()] for row in string.split('  ')])  # "  " 用于分隔行

print(array)
print(type(array))