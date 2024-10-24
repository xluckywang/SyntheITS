"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

"""Evaluation of the generated synthetic data (PCA and TSNE)"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


# Dimensionality Reduction
class DR_Visualization:
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """

    def __init__(self, ori_data, synth_data, seq_len):
        self.ori_data = ori_data
        self.synth_data = synth_data
        self.seq_len = seq_len
        self.sample_size = min([1000, len(self.ori_data)])

    def perform_dimensionality_reduction(self):
        idx = np.random.permutation(len(self.ori_data))[:self.sample_size]
        real_sample = np.asarray(self.ori_data)[idx]
        synthetic_sample = np.asarray(self.synth_data)[idx]

        # for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only
        # two componentes for both the PCA and TSNE.
        synth_data_reduced = real_sample.reshape(-1, self.seq_len)
        ori_data_reduced = np.asarray(synthetic_sample).reshape(-1, self.seq_len)

        n_components = 2
        pca = PCA(n_components=n_components)
        tsne = TSNE(n_components=n_components, n_iter=300)

        # The fit of the methods must be done only using the real sequential data
        pca.fit(ori_data_reduced)

        pca_real = pd.DataFrame(pca.transform(ori_data_reduced))
        pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

        data_reduced = np.concatenate((ori_data_reduced, synth_data_reduced), axis=0)
        tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))
        return pca_real, pca_synth, tsne_results

    def plot_scatter(self, pca_real, pca_synth, tsne_results):
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

        plt.scatter(tsne_results.iloc[:self.sample_size, 0].values, tsne_results.iloc[:self.sample_size, 1].values,
                    c='black', alpha=0.2, label='Original')
        plt.scatter(tsne_results.iloc[self.sample_size:, 0], tsne_results.iloc[self.sample_size:, 1],
                    c='red', alpha=0.2, label='Synthetic')

        ax2.legend()

        fig.suptitle('Validating synthetic vs real data diversity and distributions',
                     fontsize=16,
                     color='grey')
        plt.show()


"""
绘制真实曲线和生成曲线
"""


class Plotter:

    def __init__(self, real_data, synthetic_data):
        self.real_data = real_data
        self.synthetic_data = synthetic_data


    def plot_generated_samples(self, coles):
        fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(15, 10))
        axes = axes.flatten()

        # time = list(range(1, 25))
        obs = np.random.randint(len(self.real_data))

        for i, col in enumerate(coles):
            df = pd.DataFrame({'Real': self.real_data[obs][i, :],
                               'Synthetic': self.synthetic_data[obs][i, :]})
            df.plot(ax=axes[i], title='real VS synthetic' + '(' + 'curve' + str(i + 1) + ')',
                    secondary_y='Synthetic data', style=['-', '--'])
        fig.tight_layout()
        plt.show()

    def plot_time_series(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        axes = axes.flatten()

        for data in self.real_data:
            for curve in data:
                axes[0].plot(curve, color='blue', alpha=0.5)

        axes[0].set_title('Original Time Series Data')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')

        for data in self.synthetic_data:
            for curve in data:
                axes[1].plot(curve, color='cyan', alpha=0.5)

        axes[1].set_title('Generated Time Series Data')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Value')

        plt.tight_layout()
        plt.show()
