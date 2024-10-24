import pandas as pd
import numpy as np

# filename = f'tightening_data/original/data_ok/tightening_data_nor.csv'
filename = f'data_sets_12_24/tightening_data/original/not_ok_data.csv'

data = pd.read_csv(filename)
data = data.transpose()


num_columns = data.shape[1]
column_names = ['curve_{}'.format(i + 1) for i in range(num_columns)]  # 创建列名列表
data.columns = column_names  # 为DataFrame添加列名
# 使用melt将DataFrame转换为50x1的形式
result_df = pd.melt(data, var_name='Variable', value_name='Value')
# 将打乱后的数据保存为一个新的 CSV 文件
# data.to_csv('tightening_data/original/data_ok/tightening_data_nor_tra.csv', index=False)
result_df.to_csv('data_sets_12_24/tightening_data/original/not_ok_data_01.csv', index=False)

print("合并并随机打乱完成，并保存为文件")





