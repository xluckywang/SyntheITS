# -*- coding: utf-8 -*-
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np

# 数据:Accuracy
# data = np.array([
#     [0.9793, 0.9897, 0.9965, 0.9931, 0.9965, 0.9690, 0.9966],
#     [1.0000, 1.0000, 0.9965, 0.9965, 0.9965, 1.0000, 1.0000],
#     [0.8753, 0.3217, 0.6089, 1.0000, 0.9965, 0.9966, 0.8305],
#     [0.9793, 1.0000, 0.9965, 1.0000, 0.9965, 1.0000, 0.9966]
# ])

# 数据:F1
# data = np.array([
#     [0.9496, 0.9649, 0.9905, 0.9787, 0.9905, 0.9671, 0.9965],
#     [1.0000, 1.0000, 0.9905, 0.9905, 0.9905, 1.0000, 1.0000],
#     [0.7414, 0.3434, 0.1790, 1.0000, 0.9905, 0.9965, 0.7702],
#     [0.9496, 1.0000, 0.9905, 1.0000, 0.9905, 1.0000, 0.9965]
# ])

# # 数据:Recall
# data = np.array([
#     [1.0000, 0.9338, 1.0000, 1.0000, 1.0000, 0.9713, 0.9965],
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.9778, 0.9778, 0.1461, 1.0000, 1.0000, 0.9966, 0.8305],
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9966]
# ])

# 数据:Pre.
data = np.array([
    [0.9060, 1.0000, 1.0000, 1.0000, 1.0000, 0.9713, 0.9965],
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.5954, 0.2100, 0.1461, 1.0000, 1.0000, 0.9966, 0.8305],
    [0.9060, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9966]

])



# 列名
columns = ['Xgboost', 'Dt', 'Lda', 'Nb', 'Knn', 'Lstm', 'Resnet']

# 绘图
fig, ax = plt.subplots()
bar_width = 0.20
step = 1.00  # 调整柱状图之间的间隔

# 绘制四组柱状图，分别为Timegan、Timediffusion、doppelganger、TT
bar_positions1 = np.arange(0, step * len(columns), step) - bar_width
bar_positions2 = np.arange(0, step * len(columns), step)
bar_positions3 = np.arange(0, step * len(columns), step) + bar_width
bar_positions4 = np.arange(0, step * len(columns), step) + 2 * bar_width

bars1 = ax.bar(bar_positions1, data[0], bar_width, label='Timegan', color='#1f77b4')  # 蓝色
bars2 = ax.bar(bar_positions2, data[1], bar_width, label='Timediffusion', color='#ff7f0e')  # 橙色
bars3 = ax.bar(bar_positions3, data[2], bar_width, label='doppelganger', color='#2ca02c')  # 绿色
bars4 = ax.bar(bar_positions4, data[3], bar_width, label='TT', color='#d62728')  # 红色

# 设置刻度
ax.set_xticks(bar_positions2)
ax.set_xticklabels(columns)
ax.tick_params(axis='x', rotation=45)  # 旋转刻度标签

# 设置纵坐标刻度
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=25))
ax.yaxis.set_minor_locator(AutoMinorLocator())

# 提高纵坐标上限
ax.set_ylim(bottom=0.95, top=1.0)

# # 添加水平虚线
# ax.axhline(y=1.0, color='gray', linestyle='--')

# 添加轴标签
# ax.set_xlabel('Models')
# ax.set_ylabel('F1')
# ax.set_ylabel('Accuracy')
# ax.set_ylabel('Recall')
ax.set_ylabel('pre.')

# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=False, shadow=True, ncol=4)

# 显示图形
plt.show()
