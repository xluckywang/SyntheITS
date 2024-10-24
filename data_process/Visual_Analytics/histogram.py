# -*- coding: utf-8 -*-
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np

# 数据
data = np.array([
    [0.9640, 0.9503, 0.9612, 0.9700, 0.9713],
    [0.8212, 0.9827, 0.9612, 0.8671, 0.9865]
])

# 列名
columns = ['C1', 'C2', 'C3', 'C4', 'C5']

# 绘图
fig, ax = plt.subplots()
bar_width = 0.25
step = 0.8  # 调整柱状图之间的间隔

# 绘制两组柱状图
bar_positions = np.arange(0, step * len(columns), step)
bars1 = ax.bar(bar_positions - bar_width/2, data[0], bar_width, label='Normal')
bars2 = ax.bar(bar_positions + bar_width/2, data[1], bar_width, label='Abnormal')

# 设置刻度
ax.set_xticks(bar_positions)
ax.set_xticklabels(columns)
ax.tick_params(axis='x', rotation=45)  # 旋转刻度标签

# 设置纵坐标刻度
ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=25))
ax.yaxis.set_minor_locator(AutoMinorLocator())

# 提高纵坐标上限
ax.set_ylim(top=1.1)

# 添加水平虚线
ax.axhline(y=1.0, color='gray', linestyle='--')

# # 设置纵坐标刻度的显示
# ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

# 不显示图例
ax.legend().set_visible(False)


# 显示图形
plt.show()

