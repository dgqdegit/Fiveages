import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
# 加载 .pkl 文件
path="/data2/cyx/realworld_datasets/place_block_in_plate/0/3rd_cam_rgb/0.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

# 检查数据类型
if isinstance(data, np.ndarray):
    # 处理 numpy 数组
    image = data.copy()
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
elif torch.is_tensor(data):
    # 处理 PyTorch 张量
    image = data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
elif isinstance(data, dict):
    # 处理字典
    image = data["image"]
    # 进一步根据类型转换

# 显示图像
plt.imshow(image)
plt.axis("off")
plt.show()