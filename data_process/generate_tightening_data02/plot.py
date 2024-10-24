import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv('../data/curves_test.csv')
# data = pd.read_csv('../data/curves_data.csv')

data = pd.read_csv('curve2_nok.csv')

# 转置数据
# data = data.transpose()

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
    plt.plot(data[column], label=column)

# 显示图例
if legend:
    plt.legend()

# 显示图像
plt.show()
