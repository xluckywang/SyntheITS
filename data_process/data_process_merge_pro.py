import pandas as pd
import numpy as np

# 创建一个空的 DataFrame 来存储合并后的数据
combined_data = pd.DataFrame()

# 设置需要读取的数据文件地址变量
filename_train = f'tightening_data/original/freezer_small_train/x_train.csv'
filename_test = f'tightening_data/original/freezer_small_train/x_test.csv'

# 读取数据
data_train = pd.read_csv(filename_train)
data_test = pd.read_csv(filename_test)

# 选择 data_train 的前 14 行
subset_data_train_ok = data_train.head(14)
# 选择 data_test 的前 1425 行
subset_data_test_ok = data_test.head(1425)

# 选择 data_train 的后 14 行
subset_data_train_nok = data_train.tail(14)
# 选择 data_test 的后 1425 行
subset_data_test_nok = data_test.tail(1425)

# 合并两个 DataFrame
merged_data_ok = pd.concat([subset_data_train_ok, subset_data_test_ok], ignore_index=True)
# 合并两个 DataFrame
merged_data_nok = pd.concat([subset_data_train_nok, subset_data_test_nok], ignore_index=True)

# 随机打乱数据
merged_data_ok = merged_data_ok.sample(frac=1).reset_index(drop=True)
merged_data_nok = merged_data_nok.sample(frac=1).reset_index(drop=True)

# 将打乱后的数据保存为一个新的 CSV 文件
merged_data_ok.to_csv('tightening_data/original/freezer_small_train/freezer_small_train_ok.csv', index=False)
# 将打乱后的数据保存为一个新的 CSV 文件
merged_data_ok.to_csv('tightening_data/original/freezer_small_train/freezer_small_train_nok.csv', index=False)

print("合并并随机打乱完成，并保存为 tightening_curves_ok.csv 文件")
