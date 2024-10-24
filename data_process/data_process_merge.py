import pandas as pd
import numpy as np

# 创建一个空的 DataFrame 来存储合并后的数据
combined_data = pd.DataFrame()

# 循环读取并合并 CSV 文件
for i in range(1, 14):
    filename = f'tightening_data/generate/data_ok/curve{i}.csv'
    data = pd.read_csv(filename)

    # 将当前读取的数据按列依次合并到右侧
    combined_data = pd.concat([combined_data, data], axis=1)
combined_data = combined_data.transpose()
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)
shuffled_data = shuffled_data.transpose()

# 将打乱后的数据保存为一个新的 CSV 文件
shuffled_data.to_csv('tightening_data/generate/data_ok/tightening_curves_ok.csv', index=False)

print("合并并随机打乱完成，并保存为 tightening_curves_ok.csv 文件")
