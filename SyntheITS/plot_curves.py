import pandas as pd
import matplotlib.pyplot as plt

"""timegan"""
# data = pd.read_csv('validation_data/sony_ai_robot_data/generate/timegan/ok_data_syn.csv')
# data = pd.read_csv('validation_data/sony_ai_robot_data/original/ok_data.csv')
#
# """doppelganger"""
# data = pd.read_csv('tightening_data/generate/doppelganger/tightening_data_nor_syn.csv')
# data = pd.read_csv('tightening_data/generate/doppelganger/tightening_data_abn_syn.csv')

# data = pd.read_csv('tightening_data/generate/timediffusion/not_ok_data_syn.csv')
# data = pd.read_csv('tightening_data/original/not_ok_data.csv')

# 原始数据
# data = pd.read_csv('data_sets_12_24/tightening_data/original/not_ok_data.csv')
# data = pd.read_csv('data_sets_12_24/tightening_data/original/not_ok_data.csv')

# 生成数据
# data = pd.read_csv('data_sets_12_24/validation_data/chinatown_data/generate/timediffusion_0.99/ok_data_syn.csv')
data = pd.read_csv('data_sets_12_24/validation_data/chinatown_data/original/ok_data.csv')

#data = pd.read_csv('data_sets_12_24/validation_data/sony_ai_robot_data/generate/timegan/ok_data_syn.csv')
#data = pd.read_csv('data_sets_12_24/validation_data/sony_ai_robot_data/original/not_ok_data.csv')

# 转置数据
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
    plt.plot(data[column], label=column)

# 设置横坐标刻度间隔为50
plt.xticks(range(0, len(data[column]), 2))

# 显示图例
if legend:
    plt.legend()

# 显示图像
plt.show()
