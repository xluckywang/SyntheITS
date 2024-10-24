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
output_directory = '../data_sets_12_24/tightening_data/generate/timegan'

# 指定日志文件路径，保存在输出文件夹的父目录下
log_file_path = os.path.join(output_directory, 'program_log.txt')

# 配置日志
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 1. 记录整体程序的开始时间
total_start_time = time.time()

# 2. 定义模型参数和训练参数
gan_args = ModelParameters(batch_size=32, lr=5e-4, noise_dim=32, layers_dim=128, latent_dim=24, gamma=1)
train_args = TrainParameters(epochs=1000, sequence_length=6, number_sequences=500)

# 3. 读取数据
read_data_path = '../data_sets_12_24/tightening_data/original/ok_data.csv'
save_data_path = '../data_sets_12_24/tightening_data/generate/timegan/ok_data_syn.csv'

tightening_data_ok = pd.read_csv(read_data_path)
tightening_data_ok = tightening_data_ok.sample(frac=1, random_state=42)  # 随机打乱行元素
tightening_data_ok = tightening_data_ok.reset_index(drop=True)  # 重置索引
cols = list(tightening_data_ok.columns)

# 4. 处理模型保存路径
model_save_path = '../data_sets_12_24/tightening_data/generate/timegan/synthesizer_tightening_ok_1000.pkl'
model_exists = path.exists(model_save_path)

train_start_time = 0
train_end_time = 0

# 5. 训练或加载模型
if model_exists:
    synth = TimeSeriesSynthesizer.load(model_save_path)
else:
    # 记录模型训练的开始时间
    train_start_time = time.time()
    synth = TimeSeriesSynthesizer(modelname='timegan', model_parameters=gan_args)
    synth.fit(tightening_data_ok, train_args, num_cols=cols)
    synth.save(model_save_path)
    # 记录模型训练的结束时间
    train_end_time = time.time()

# 6. 处理原始数据
tightening_data_ok_blocks, scaler = processed_stock(path=read_data_path, seq_len=6)

# 7. 生成合成样本
# 记录模型测试的开始时间
test_start_time = time.time()
synth_data_list = synth.sample(n_samples=len(tightening_data_ok_blocks))
# 记录模型测试的结束时间
test_end_time = time.time()

synth_data_array = np.array(synth_data_list)

# 8. 调整维度
n_samples, seq_len, n_features = synth_data_array.shape
synth_data_reshaped = synth_data_array.reshape((n_samples * seq_len, n_features))

# 9. 逆向归一化并保留八位小数
synth_data_original = np.round(scaler.inverse_transform(synth_data_reshaped), 1)

# 10. 将逆向归一化后的数据转换为 DataFrame
synth_data_df = pd.DataFrame(data=synth_data_original, columns=cols)

# 11. 打印 DataFrame 的形状
print(synth_data_df.shape)
print(synth_data_list[0].shape)

synth_data_df.to_csv(save_data_path, index=False)

# 12.记录整体程序的结束时间
total_end_time = time.time()

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
