import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


def recurrence_plot(data, threshold=0.1):
    """
    Generate a recurrence plot from a time series.

    :param data: Time series data
    :param threshold: Threshold to determine recurrence
    :return: Recurrence plot
    """
    # Calculate the distance matrix
    N = len(data)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.abs(data[i] - data[j])
            # distance_matrix[i, j] = np.abs(np.asarray(data[i]) - np.asarray(data[j]))

            # Create the recurrence plot
    recurrence_plot = np.where(distance_matrix <= threshold, 1, 0)

    return recurrence_plot


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

"""
Creating sequence data
"""
history = pd.read_csv("../../data/original_data/tightening_data_nor.csv")  # .values
# history = history.transpose()
num_columns = history.shape[1]
column_names = ['curve{}'.format(i + 1) for i in range(num_columns)]  # 创建列名列表
history.columns = column_names  # 为DataFrame添加列名
seq = history.curve1.values  # 将历史数据的收盘价转换为一个行向

plt.figure(figsize=(10, 6))
plt.plot(seq, label='Torque curve')
plt.title('Tightening Data Time Series')
plt.xlabel('Time')
plt.ylabel('Torque')
plt.legend()
plt.grid(True)
plt.show()

"""递归图为这种白噪声提供了有趣的可视化效果。"""
# Generate and plot the recurrence plot
recurrence = recurrence_plot(seq, threshold=0.1)

plt.figure(figsize=(8, 8))
plt.imshow(recurrence, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time')
plt.ylabel('Time')
plt.colorbar(label='Recurrence')
plt.show()
