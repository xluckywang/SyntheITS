import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


def calculate_kl_divergence(chinatown_data, synth_data):
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

    # 将概率值中的0替换为一个较小的非零值，避免计算问题
    p[p == 0] = np.finfo(float).eps
    q[q == 0] = np.finfo(float).eps

    # 计算KL散度
    kl_divergence = np.sum(chinatown_dist * np.log(p / q), axis=1)

    # 输出KL散度结果
    return kl_divergence
