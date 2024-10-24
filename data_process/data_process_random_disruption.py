import pandas as pd
import numpy as np

# 循环读取并合并 CSV 文件

filename = f'tightening_data/original/timegan_40_1000pkl/tightening_data_abn_syn.csv'
data = pd.read_csv(filename)

shuffled_data = data.sample(frac=1).reset_index(drop=True)

# 将打乱后的数据保存为一个新的 CSV 文件
shuffled_data.to_csv('tightening_data/original/timegan_40_1000pkl/tightening_data_abn_syn_disruption.csv', index=False)

print("合并并随机打乱完成，并保存为 tightening_data_abn_syn_disruption.csv 文件")
