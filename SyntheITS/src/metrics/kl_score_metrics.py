import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


def calculate_kl_divergence_01(chinatown_data, synth_data, inverse_normalized_data):
    # 将数据转换为numpy数组
    chinatown_data = np.array(chinatown_data)
    synth_data = np.array(synth_data)

    # 将数据展平为向量
    chinatown_data_flat = chinatown_data.reshape(chinatown_data.shape[0], -1)
    synth_data_flat = synth_data.reshape(synth_data.shape[0], -1)

    # 计算概率分布
    chinatown_dist = chinatown_data_flat / np.sum(chinatown_data_flat, axis=1, keepdims=True)
    synth_dist = synth_data_flat / np.sum(synth_data_flat, axis=1, keepdims=True)

    p = chinatown_dist
    q = synth_dist

    # 将概率值进行平滑处理，避免计算问题
    eps = np.finfo(float).eps
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    # 计算KL散度
    kl_divergence_01 = np.sum(chinatown_dist * np.log(p / q), axis=1)

    # 筛选散度较小的数据
    filtered_indices = np.where(kl_divergence_01 < 0.15)[0]
    filtered_chinatown_data = chinatown_data[filtered_indices]
    filtered_synth_data = inverse_normalized_data[filtered_indices]

    # 输出KL散度结果
    return kl_divergence_01, filtered_chinatown_data, filtered_synth_data

"""
def calculate_kl_divergence_02(chinatown_data_, synth_data_, num_components):
    # 将数据转换为numpy数组
    chinatown_data_ = np.array(chinatown_data_)
    synth_data_ = np.array(synth_data_)

    # 将数据展平为二维数组
    chinatown_data_ = chinatown_data_.reshape(-1, chinatown_data_.shape[-1])
    synth_data_ = synth_data_.reshape(-1, synth_data_.shape[-1])

    # 合并原始数据和生成数据
    combined_data = np.concatenate((chinatown_data_, synth_data_), axis=0)

    # 拟合数据的GMM模型
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(combined_data)

    # 计算每个样本属于每个组件的概率
    p_ = gmm.predict_proba(chinatown_data_)
    q_ = gmm.predict_proba(synth_data_)

    # # 计算每个样本行概率分布
    # p_1 = np.sum(p_, axis=1, keepdims=True)
    # q_1 = np.sum(q_, axis=1, keepdims=True)
    #
    # # 计算每个样本行概率分布
    # p_11 = p_1 / np.sum(p_1, axis=0, keepdims=True)
    # q_11 = q_1 / np.sum(q_1, axis=0, keepdims=True)

    # 将概率值进行平滑处理，避免计算问题
    eps = np.finfo(float).eps
    p_11 = np.maximum(p_11, eps)
    q_11 = np.maximum(q_11, eps)

    # 计算KL散度
    kl_divergence_02 = np.sum(p_11 * np.log(p_11 / q_11))

    return kl_divergence_02
"""
# # 计算每列之和和每行之和
# column_sums = chinatown_data_.sum(axis=0)
# row_sums = chinatown_data_.sum(axis=1)
#
# # 打印结果
# print("Column sums:")
# print(column_sums)
# print("\nRow sums:")
# print(row_sums)
