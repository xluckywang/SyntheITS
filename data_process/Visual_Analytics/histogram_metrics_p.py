# -*- coding: utf-8 -*-
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np

# # # 指标：Timegan
# lstm
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.8581, 0.9857, 0.9697, 0.9714, 0.9714]
# ])

# # resnet
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 0.9990, 1.0000],
#     [0.8476, 0.9436, 0.9492, 0.9216, 0.9451]
# ])

# lstm
data = np.array([
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    [0.8581, 0.9857, 0.9697, 0.9714, 0.9714]
])



# # # 指标：Timegan
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.8581, 0.9857, 0.9697, 0.9714, 0.9714]
# ])

# # # 指标：Timegan
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.8581, 0.9857, 0.9697, 0.9714, 0.9714]
# ])
#
# # # 指标：Timegan
# data = np.array([
#     [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#     [0.8581, 0.9857, 0.9697, 0.9714, 0.9714]
# ])

# 列名
columns = ['0.00', '0.25', '0.50', '0.75', '1.00']

# 绘图
fig, ax = plt.subplots()
bar_width = 0.20
step = 0.6  # 调整柱状图之间的间隔

# 绘制两组柱状图
bar_positions = np.arange(0, step * len(columns), step)
bars1 = ax.bar(bar_positions - bar_width / 2, data[0], bar_width, label='Normal')
bars2 = ax.bar(bar_positions + bar_width / 2, data[1], bar_width, label='Abnormal')

# 设置刻度
ax.set_xticks(bar_positions)
ax.set_xticklabels(columns)
ax.tick_params(axis='x', rotation=45)  # 旋转刻度标签

# 设置纵坐标刻度
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
ax.yaxis.set_minor_locator(AutoMinorLocator())

# 提高纵坐标上限
ax.set_ylim(bottom=0.80, top=1.0)

# # 添加水平虚线
# ax.axhline(y=1.0, color='gray', linestyle='--')

# 添加轴标签
# ax.set_xlabel('Models')
ax.set_ylabel('Probability')



# 显示图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=False, shadow=True, ncol=4)

# 显示图形
plt.show()
