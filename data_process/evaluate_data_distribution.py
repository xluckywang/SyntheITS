# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

synth_data_reduced = pd.read_csv('data_sets_12_24/validation_data/chinatown_data/generate/ok_data_syn.csv')#.transpose()
stock_data_reduced = pd.read_csv('data_sets_12_24/validation_data/chinatown_data/original/ok_data.csv')#.transpose()

# 将数据转换为 NumPy 数组
synth_data_reduced = synth_data_reduced.values
stock_data_reduced = stock_data_reduced.values

# 创建 StandardScaler 对象
scaler = StandardScaler()

# 对合成数据进行标准化
synth_data_reduced = scaler.fit_transform(synth_data_reduced)

# 对原始数据进行标准化
stock_data_reduced = scaler.fit_transform(stock_data_reduced)

sample_size = 500
# idx = np.random.permutation(len(stock_data))[:sample_size]
#
# real_sample = np.asarray(stock_data)[idx]
# synthetic_sample = np.asarray(synth_data)[idx]
#
# # for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
# synth_data_reduced = real_sample.reshape(-1, seq_len)
# stock_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)

n_components = 2
pca = PCA(n_components=n_components)
tsne = TSNE(n_components=n_components, n_iter=300)

# The fit of the methods must be done only using the real sequential data
pca.fit(stock_data_reduced)

pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

# TSNE scatter plot
ax = fig.add_subplot(spec[0, 0])
ax.set_title('PCA results',
             fontsize=20,
             color='red',
             pad=10)

# PCA scatter plot
plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
            c='black', alpha=0.2, label='Original')
plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
            c='red', alpha=0.2, label='Synthetic')
ax.legend()

ax2 = fig.add_subplot(spec[0, 1])
ax2.set_title('TSNE results',
              fontsize=20,
              color='red',
              pad=10)

plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
            c='black', alpha=0.2, label='Original')
plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
            c='red', alpha=0.2, label='Synthetic')

ax2.legend()

fig.suptitle('Validating synthetic vs real data diversity and distributions',
             fontsize=16,
             color='grey')
plt.show()  # Added to display the plot
