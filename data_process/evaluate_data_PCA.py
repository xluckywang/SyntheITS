# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 导入数据
original_path = 'data_sets_01_15/tightening_data/original/ok_data.csv'
generated_path = 'data_sets_01_15/tightening_data/generate/timediffusion/ok_data_syn.csv'

df_generated = pd.read_csv(generated_path)
df_original = pd.read_csv(original_path)

# 标准化数据
scaler = StandardScaler()
scaled_original = scaler.fit_transform(df_original)
scaled_generated = scaler.transform(df_generated)

# 应用PCA
pca = PCA()
pca.fit(scaled_original)

# 查看解释方差比例
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance_ratio}")

# 对生成的数据应用相同的PCA转换
transformed_generated = pca.transform(scaled_generated)

# 比较主成分
# 选择前两个主成分进行比较
plt.scatter(transformed_generated[:, 0], transformed_generated[:, 1], label='Generated Data', marker='o')
plt.scatter(pca.transform(scaled_original)[:, 0], pca.transform(scaled_original)[:, 1], label='Original Data', marker='o')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Comparison of Original and Generated Data')
plt.show()
