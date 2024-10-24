import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import pandas as pd
import os


def recurrence_plot(data, threshold=0.1):
    # 从时间序列生成递归图
    data_length = len(data)
    distance_matrix = np.abs(np.subtract.outer(data, data))
    recurrence_plots = np.where(distance_matrix <= threshold, 1, 0)
    return recurrence_plots


def plot_torque_curve(seq_, title='Tightening Data Time Series'):
    # 绘制扭矩曲线
    plt.plot(seq_, label='Torque curve')
    setup_plot(title, 'Time', 'Torque')


def plot_recurrence(data, threshold=0.1, title='Recurrence Plot'):
    # 生成并绘制递归图
    recurrence = recurrence_plot(data, threshold)
    plt.imshow(recurrence, cmap='binary', origin='lower', interpolation='nearest')
    setup_plot(title, 'Time', 'Time', legend=False, color='binary')


def setup_plot(title, xlabel, ylabel, legend=True, color='blue'):
    # 设置图形属性的通用函数
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.grid(True)


# 设置设备
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA（GPU）可用。")
else:
    device = "cpu"
    print("CUDA（GPU）不可用。使用CPU.")

# 文件路径作为参数传递
# file_path = os.path.join("..", "..", "data", "original_data", "tightening_data_nor.csv")
file_path = os.path.join("..", "..", "data", "original_data", "tightening_data_abn.csv")
# 读取数据
history = pd.read_csv(file_path)
num_columns = history.shape[1]
column_names = [f'curve{i + 1}' for i in range(num_columns)]
history.columns = column_names

# 设置动画
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))


def update(frame):
    # 每一帧更新图形
    ax1.clear()
    ax2.clear()

    n = frame % num_columns
    curve_num = f"curve{n + 1}"
    seq = history[curve_num].values

    # 绘制扭矩曲线
    ax1.plot(seq, label='Torque curve')
    setup_plot(f'Tightening Data Time Series - {curve_num}', 'Time', 'Torque', legend=False)

    # 绘制递归图
    recurrence = recurrence_plot(seq, threshold=0.5)
    ax2.imshow(recurrence, cmap='binary', origin='lower', interpolation='nearest')
    setup_plot(f'Recurrence Plot - {curve_num}', 'Time', 'Time', legend=False, color='binary')


# 设置动画帧数
num_frames = num_columns
ani = FuncAnimation(fig, update, frames=num_frames, repeat=True)

plt.show()
