import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


def recurrence_plot(data, threshold=0.1):
    """
    从时间序列生成一个递归图。

    :param data: np.ndarray或list，时间序列数据
    :param threshold: float，用于确定递归的阈值
    :return: np.ndarray，递归图
    """

    # Calculate the distance matrix
    data_length = len(data)
    distance_matrix = np.abs(np.subtract.outer(data, data))

    # Create the recurrence plot
    recurrence_plots = np.where(distance_matrix <= threshold, 1, 0)

    return recurrence_plots


def plot_torque_curve(seq_, title='Tightening Data Time Series'):
    """
    Plot the torque curve.

    :param seq_: Torque data sequence
    :param title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(seq_, label='Torque curve')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Torque')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_recurrence(data, threshold=0.1, title='Recurrence Plot'):
    """
    Generate and plot the recurrence plot.

    :param data: Time series data
    :param threshold: Threshold for recurrence
    :param title: Title for the plot
    """
    # Generate recurrence plot
    recurrence = recurrence_plot(data, threshold)

    # Plot recurrence plot
    plt.figure(figsize=(8, 8))
    plt.imshow(recurrence, cmap='binary', origin='lower', interpolation='nearest')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.colorbar(label='Recurrence')
    plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 文件路径作为参数传递
file_path = "../../data/original_data/tightening_data_nor.csv"

"""
Creating sequence data
"""
history = pd.read_csv(file_path)
num_columns = history.shape[1]
column_names = [f'curve{i + 1}' for i in range(num_columns)]
history.columns = column_names

for n in range(num_columns):
    curve_num = f"curve{n + 1}"
    seq = history[curve_num].values

    # 调用方法进行可视化
    plot_torque_curve(seq, title=f'Tightening Data Time Series - {curve_num}')
    # 调用方法进行递归图可视化
    plot_recurrence(seq, threshold=0.5, title=f'Recurrence Plot - {curve_num}')
