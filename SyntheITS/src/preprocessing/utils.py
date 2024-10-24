"""
    Utility functions to be shared by the time-series preprocessing required to feed the data into the synthesizers
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Method implemented here: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
# Originally used in TimeGAN research
def real_data_loading(data: np.array, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    # 将数据翻转以保持时间顺序
    ori_data = data[::-1]
    # 对数据进行归一化
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # 预处理数据集
    temp_data = []

    # 根据序列长度切分数据
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # 混合数据集（使其类似于独立同分布）
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data, scaler
