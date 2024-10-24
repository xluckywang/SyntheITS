# -*- coding: utf-8 -*-
import pandas as pd

# 读取训练集和测试集
train_data = pd.read_csv('data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_train.csv')
test_data = pd.read_csv('data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_test.csv')

num_columns = train_data.shape[1]
column_names = [f'curve{i + 1}' for i in range(num_columns)]

train_data.columns = column_names
test_data.columns = column_names

# 获取最后一列的列名
label_column = train_data.columns[-1]

# 分离正样本和负样本
positive_samples_train = train_data[train_data[label_column] == 1]
negative_samples_train = train_data[train_data[label_column] == 0]

positive_samples_test = test_data[test_data[label_column] == 1]
negative_samples_test = test_data[test_data[label_column] == 0]

# 打印结果或进行其他处理
print("训练集中的正样本：\n", positive_samples_train)
print("训练集中的负样本：\n", negative_samples_train)

print("测试集中的正样本：\n", positive_samples_test)
print("测试集中的负样本：\n", negative_samples_test)

# 合并正样本和负样本
positive_samples_combined = pd.concat([positive_samples_train, positive_samples_test], axis=0).iloc[:, :-1]
negative_samples_combined = pd.concat([negative_samples_train, negative_samples_test], axis=0).iloc[:, :-1]

# 保存合并后的数据为CSV文件
positive_samples_combined.to_csv('data_sets_12_24/validation_data/share_price_in_crease_data/sample_total/share_price_in_crease_data_nok.csv', index=False)
negative_samples_combined.to_csv('data_sets_12_24/validation_data/share_price_in_crease_data/sample_total/share_price_in_crease_data_ok.csv', index=False)
