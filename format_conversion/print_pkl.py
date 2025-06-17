import pickle
import pandas as pd

# 定义一个函数来加载和打印文件内容
def load_and_print_pkl(file_path):
    # 打开 pkl 文件并加载数据
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # 打印数据类型
    print(f'数据类型: {type(data)}')
    
    # 如果数据是 DataFrame 或 ndarray，打印形状
    if isinstance(data, pd.DataFrame):
        print(f'数据形状: {data.shape}')
    elif isinstance(data, (list, tuple)):
        print(f'数据长度: {len(data)}')
    elif hasattr(data, 'shape'):
        print(f'数据形状: {data.shape}')
    else:
        print('数据没有形状属性')
    
    # 打印数据内容的前几行
    print('数据内容:')
    print(data)

# 使用函数并传入 pkl 文件路径
load_and_print_pkl('/home/zk/Projects/Datasets/put_shelf_0611_right/0/actions/0.pkl')

