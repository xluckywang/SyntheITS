# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 导入数据
original_path = 'data_sets_12_24/tightening_data/original/ok_data.csv'
generated_path = 'data_sets_12_24/tightening_data/generate/timediffusion/ok_data_syn.csv'

df_generated = pd.read_csv(generated_path)
df_original = pd.read_csv(original_path)

# 标准化数据
scaler = StandardScaler()
scaled_original = scaler.fit_transform(df_original)
scaled_generated = scaler.transform(df_generated)

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42)
transformed_generated = tsne.fit_transform(scaled_generated)
transformed_original = tsne.fit_transform(scaled_original)

# 绘制散点图，比较原始数据和生成数据在 t-SNE 空间的分布情况
plt.scatter(transformed_generated[:, 0], transformed_generated[:, 1], label='Generated Data', marker='o')
plt.scatter(transformed_original[:, 0], transformed_original[:, 1], label='Original Data', marker='o')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.title('Comparison of Original and Generated Data using t-SNE')
plt.show()
