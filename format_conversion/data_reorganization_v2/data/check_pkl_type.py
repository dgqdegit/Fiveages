import pickle

# 加载 .pkl 文件
file_path = '/home/zk/Projects/DobotStudio/vla_data/data/data/right_wbl/0/extrinsic_matrix.pkl'  # 替换为您的 .pkl 文件路径

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 输出加载的数据类型
print(type(data))
