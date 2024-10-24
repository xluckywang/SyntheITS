# -*- coding: utf-8 -*-
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np

# # 指标：F1
# data = np.array([
#     [0.9496, 0.9496, 0.9496, 0.9496],
#     [0.9496, 1.0000, 0.9905, 0.9905],
#     [0.9821, 1.0000, 0.9660, 0.9821],
#     [0.9467, 1.0000, 0.9905, 0.9895],
#     [0.9496, 1.0000, 0.9496, 0.9895]
#
# ])

# 指标：Accuracy
# data = np.array([
#     [0.9793, 0.9793, 0.9793, 0.9793],
#     [0.9793, 1.0000, 0.9965, 0.9965],
#     [0.9931, 1.0000, 0.9862, 0.9931],
#     [0.9827, 1.0000, 0.9965, 0.9966],
#     [0.9793, 1.0000, 0.9793, 0.9966]
# ])


# # 指标：Recall
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000],
#     [1.0000, 1.0000, 0.9818, 0.9818],
#     [1.0000, 1.0000, 1.0000, 1.0000],
#     [0.9003, 1.0000, 0.9818, 1.0000],
#     [1.0000, 1.0000, 1.0000, 1.0000]
# ])

# 指标：pre.
data = np.array([
    [0.9060, 0.9060, 0.9060, 0.9060],
    [0.9060, 1.0000, 1.0000, 1.0000],
    [0.9657, 1.0000, 0.9348, 0.9657],
    [1.0000, 1.0000, 1.0000, 0.9800],
    [0.9060, 1.0000, 0.9060, 0.9800]
])

# 对数据进行转置
data = np.transpose(data)

# 列名
columns = ['0.00', '0.25', '0.50', '0.75', '1.00']

# 绘图
fig, ax = plt.subplots()
bar_width = 0.20
step = 1.20  # 调整柱状图之间的间隔

# 绘制四组柱状图，分别为Timegan、Timediffusion、doppelganger、TT
bar_positions1 = np.arange(0, step * len(columns), step) - bar_width
bar_positions2 = np.arange(0, step * len(columns), step)
bar_positions3 = np.arange(0, step * len(columns), step) + bar_width
bar_positions4 = np.arange(0, step * len(columns), step) + 2 * bar_width
# bar_positions5 = np.arange(0, step * len(columns), step) + 3 * bar_width

bars1 = ax.bar(bar_positions1, data[0], bar_width, label='timegan', color='#1f77b4')  # 蓝色
bars2 = ax.bar(bar_positions2, data[1], bar_width, label='TimeDiffusion', color='#ff7f0e')  # 橙色
bars3 = ax.bar(bar_positions3, data[2], bar_width, label='DoppelGANger', color='#2ca02c')  # 绿色
bars4 = ax.bar(bar_positions4, data[3], bar_width, label='TT', color='#d62728')  # 红色color='#d62728'
# bars4 = ax.bar(bar_positions5, data[4].TT, bar_width, label='1.00', color='#17becf')  # 红色

# 设置刻度
ax.set_xticks(bar_positions3)
ax.set_xticklabels(columns)
ax.tick_params(axis='x', rotation=0)  # 旋转刻度标签

# 设置纵坐标刻度
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
ax.yaxis.set_minor_locator(AutoMinorLocator())

# 提高纵坐标上限
ax.set_ylim(bottom=0.90, top=1.0)

# # 添加水平虚线
# ax.axhline(y=1.0, color='gray', linestyle='--')

# 添加轴标签
# ax.set_xlabel('Models')
# ax.set_ylabel('F1')
ax.set_ylabel('Accuracy')
# ax.set_ylabel('Recall')
# ax.set_ylabel('pre.')

# # # 设置纵坐标刻度的显示
# ax.set_yticks([0.7, 0.8, 1.0])
# ax.set_yticklabels([0.7, 0.8, 1.0])

# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=False, shadow=True, ncol=4)

# 显示图形
plt.show()
