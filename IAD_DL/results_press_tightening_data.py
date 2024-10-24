# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np  # Import NumPy for vstack

# 自定义保存基础路径
base_path_images = r'experimental_results/tightening_data/'

# 设置数据生成模型类型
synthesizer_model_type = 'timegan_and_timediffusion'
# doppelganger\timediffusion\timegan\timegan_and_timediffusion
# models_name = ["resnet"]
models_name = ["lstm", "resnet"]

# data_type = 'chinatown_data'
ratio = [0.00, 0.25, 0.50, 0.75, 1.00]
metrics_name = ["accuracy", "precision", "recall", "f1", "app", "anp"]

for mo_name in models_name:
    print(mo_name)
    # 打印空行
    print("\n" * 1)
    for j, me_name in enumerate(metrics_name):
        values_list = []
        for i, ratio_value in enumerate(ratio):
            # 设置保存路径
            images_save_path = os.path.normpath(os.path.join(base_path_images,
                                                             synthesizer_model_type,
                                                             f'experiment_{ratio_value:.2f}',
                                                             f'model_{mo_name}',
                                                             'test_result.csv'))

            # 读取 CSV 文件
            df = pd.read_csv(images_save_path, encoding='latin-1')

            # 获取 me_name 列的前 4 行数据，并进行转置
            transposed_values = df[me_name].iloc[:5].values.reshape(1, -1)
            values_list.append(transposed_values)

        # 使用 ratio 列表作为行名，创建 DataFrame
        df = pd.DataFrame(np.vstack(values_list), index=[f'ratio_{m:.2f}' for m in ratio],
                          columns=[f'test_{n + 1}' for n in range(len(ratio))])

        # 使用 round 函数保留4位小数
        df = df.round(4)

        # 对 DataFrame 进行转置，然后使用 describe 函数
        row_stats = df.transpose().describe().transpose()
        # 使用 round 函数保留4位小数
        row_stats = row_stats.round(4)

        # 使用 concat 函数在行的方向进行合并，axis=0 表示按行合并
        # merged_df = pd.concat([df, row_stats], axis=1)

        # 保存合并后的 DataFrame 到 CSV 文件
        # merged_df.to_csv(f'../data_sets/{mo_name}.csv', index=False)

        print(me_name)
        # 打印 DataFrame
        print(df)
        # 打印每行的均值、标准差、最小值、25% 分位数、中位数、75% 分位数和最大值
        print(row_stats)
        # 打印空行
        print("\n" * 2)
