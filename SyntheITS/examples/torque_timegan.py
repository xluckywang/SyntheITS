"""
    TimeGAN architecture example file
"""

# Importing necessary libraries
from os import path
from synthesizers import TimeSeriesSynthesizer
from preprocessing import processed_stock
from synthesizers import ModelParameters, TrainParameters
import numpy as np
import pandas as pd

# Define model parameters
gan_args = ModelParameters(batch_size=32,  # GAN模型的批次大小
                           lr=5e-4,  # 学习率
                           noise_dim=32,  # 噪声维度
                           layers_dim=128,  # 模型层维度
                           latent_dim=24,  # 潜变量维度
                           gamma=1)  # 生成器的损失权重

train_args = TrainParameters(epochs=100,  # 训练轮数 TD epochs=50000
                             sequence_length=5,  # 序列长度
                             number_sequences=500)  # 序列数量

# Read the data
tightening_data_ok = pd.read_csv('../data_sets/original/tightening_data_nor_tra.csv')  # 读取正常原始扭矩数据文件
# tightening_data_nok = pd.read_csv('../data_sets/original/tightening_data_abn_tra.csv')  # 读取异常原始扭矩数据文件

cols = list(tightening_data_ok.columns)  # 获取列名列表

# Training the TimeGAN synthesizer
if path.exists('synthesizer_tightening_ok_1000.pkl'):  # 判断是否已有保存的模型
    synth = TimeSeriesSynthesizer.load('synthesizer_tightening_ok_1000.pkl')  # 加载已有的模型
else:
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)  # 创建TimeSeriesSynthesizer对象
    synth.fit(tightening_data_ok, train_args, num_cols=cols)  # 训练模型
    synth.save('synthesizer_tightening_ok_1000.pkl')  # 保存训练好的模型

# Generating new synthetic samples
tightening_data_ok_blocks, scaler = processed_stock(
    path='../data_sets/original/tightening_data_nor_tra.csv', seq_len=5)  # 对原始数据进行处理，得到序列块数据

synth_data_list = synth.sample(n_samples=len(tightening_data_ok_blocks))  # 生成与原始数据相同数量的合成样本
# 将列表转换为 NumPy 数组
synth_data_array = np.array(synth_data_list)
# 假设 synth_data 的形状为 (n_samples, seq_len, n_features)
n_samples, seq_len, n_features = synth_data_array.shape
# 调整维度
synth_data_reshaped = synth_data_array.reshape((n_samples * seq_len, n_features))

# 逆向归一化，将 synth_data 恢复到原始值域
synth_data_original = scaler.inverse_transform(synth_data_reshaped)
print(synth_data_list[0].shape)



