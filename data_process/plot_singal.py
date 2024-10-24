import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('tightening_data/original/ok_data.csv')
data = data.transpose()
# 获取列数
num_columns = len(data.columns)

# 创建列名列表
column_names = list(data.columns)

# 为数据帧添加列名
data.columns = column_names

# 设置图例
if num_columns <= 10:
    legend = True
else:
    legend = False

# 绘制每一列的图像
for column in column_names:
    # plt.plot(data[1], label=column)
    plt.plot(data[5], linewidth=10, color='purple')

# 显示图例
if legend:
    plt.legend()

# 隐藏横坐标
plt.xticks([])

# 显示图像
plt.show()
