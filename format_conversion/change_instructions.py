import pickle
import os

# 替换为你的路径
parent_folder = "/home/zk/Projects/Datasets/put_the_bottle_in_the_microwave_25_0604"

# 新内容：可替换为你需要写入的数据
new_content = "put the lemonade in the microwave"


for i in range(5, 10):  # 2, 3, 4, 5
    folder_path = os.path.join(parent_folder, str(i))
    pkl_path = os.path.join(folder_path, "instruction.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "wb") as f:
            pickle.dump(new_content, f)
        print(f"已更新: {pkl_path}")
    else:
        print(f"文件不存在: {pkl_path}")
