# Importing necessary libraries
import os
import time
import logging
import numpy as np
import pandas as pd
import os.path as path
from datetime import datetime
from preprocessing import processed_stock
from synthesizers import TimeSeriesSynthesizer, ModelParameters, TrainParameters

# 获取当前时间的字符串表示
start_current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 指定日志保存文件夹路径
output_directory = '../data_sets_12_24/tightening_data/generate/doppelganger_test'
# output_directory = '../data_sets_12_24/validation_data/chinatown_data/generate/doppelganger_test'

# 指定日志文件路径，保存在输出文件夹的父目录下
log_file_path = os.path.join(output_directory, 'program_log.txt')

# 配置日志
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 1. 记录整体程序的开始时间
total_start_time = time.time()

# 2. 设置读取、存储数据地址
read_data_path = '../data_sets_12_24/tightening_data/original/ok_data.csv'
save_data_path = '../data_sets_12_24/tightening_data/generate/doppelganger_test/ok_data_syn.csv'
# read_data_path = '../data_sets_12_24/validation_data/chinatown_data/original/not_ok_data.csv'
# save_data_path = '../data_sets_12_24/validation_data/chinatown_data/generate/doppelganger_test/not_ok_data_syn.csv'

# 3. 读取数据数据，获取列名字
tightening_data_ok = pd.read_csv(read_data_path, encoding='utf-8')  # 读取正常原始扭矩数据文件, encoding='utf-8'
tightening_data_ok = tightening_data_ok.sample(frac=1, random_state=42)  # 随机打乱行元素
tightening_data_ok = tightening_data_ok.reset_index(drop=True)  # 重置索引
cols = list(tightening_data_ok.columns)  # 获取列名列表

# 4. 处理模型保存路径
model_save_path = 'doppelganger_synthesizer_tightening_ok_100.pkl'
# model_save_path = '../data_sets_12_24/tightening_data/generate/doppelganger_test' \
#                   '/doppelganger_synthesizer_chinatown_not_ok_100.pkl'
# model_save_path = '../data_sets_12_24/validation_data/chinatown_data/generate/doppelganger_test' \
#                    '/doppelganger_synthesizer_chinatown_not_ok_100.pkl'
model_exists = path.exists(model_save_path)

# 5. 处理原始数据
tightening_data_ok_blocks, scaler = processed_stock(path=read_data_path, seq_len=6)  # 对原始数据进行处理，得到序列块数据
tightening_data_ok = [pd.DataFrame(sd, columns=cols) for sd in tightening_data_ok_blocks]
tightening_data_ok = pd.concat(tightening_data_ok).reset_index(drop=True)

train_start_time = 0
train_end_time = 0

# 6. 定义模型参数和训练参数
model_args = ModelParameters(batch_size=6,
                             lr=0.001,
                             betas=(0.5, 0.9),
                             latent_dim=3,
                             gp_lambda=10,
                             pac=10)

train_args = TrainParameters(epochs=100,
                             sequence_length=6,
                             measurement_cols=cols)

# 7. 训练或加载模型
if model_exists:
    model_dop_gan = TimeSeriesSynthesizer.load(model_save_path)
else:
    # 记录模型训练的开始时间
    train_start_time = time.time()
    model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=model_args)
    model_dop_gan.fit(tightening_data_ok, train_args, num_cols=cols)  # 训练模型
    model_dop_gan.save(model_save_path)
    # 记录模型训练的结束时间
    train_end_time = time.time()

# 8. 生成合成样本
test_start_time = time.time()  # 记录模型测试的开始时间
synth_data_list = model_dop_gan.sample(n_samples=tightening_data_ok_blocks)  # 生成与原始数据数量相同的数据
test_end_time = time.time()  # 记录模型测试的结束时间
synth_data_array = np.array(synth_data_list)  # 将列表转换为 NumPy 数组

# 9. 调整维度
n_samples, seq_len, n_features = synth_data_array.shape
synth_data_reshaped = synth_data_array.reshape((n_samples * seq_len, n_features))

# 10. 逆向归一化并保留一位小数
synth_data_original = np.round(scaler.inverse_transform(synth_data_reshaped), 1)

print(synth_data_list[0])

# 11. 将逆向归一化后的数据转换为 DataFrame
synth_data_df = pd.DataFrame(data=synth_data_original, columns=cols)

# 12. 打印 DataFrame 的形状，并保存数据
print(synth_data_df.shape)
print(synth_data_list[0].shape)

synth_data_df.to_csv(save_data_path, index=False)

# 13.记录整体程序的结束时间
total_end_time = time.time()

# 计算程序运行时间
train_elapsed_time = train_end_time - train_start_time
test_elapsed_time = test_end_time - test_start_time
total_elapsed_time = total_end_time - total_start_time

# 获取当前时间的字符串表示
end_current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 时间写入日志
with open(log_file_path, 'a') as log_file:
    log_file.write(f'开始日期: {start_current_time}\n')
    log_file.write(f'结束日期: {end_current_time}\n')
    log_file.write(f'存储路径: {save_data_path}\n')
    log_file.write(f"训练用时: {train_elapsed_time:.2f} 秒\n")
    log_file.write(f"测试用时: {test_elapsed_time:.2f} 秒\n")
    log_file.write(f"整体程序用时: {total_elapsed_time:.2f} 秒\n")
