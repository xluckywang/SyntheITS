import os
import time
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import torch
from src.synthesizers.timediffusion import TD

"""
版本01：
曲线单条输入
"""

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f'Device in use: {device}')

# 文件路径配置
file_path = "../data_sets_12_24/validation_data/chinatown_data/original/not_ok_data.csv"
output_directory = '../data_sets_12_24/validation_data/chinatown_data/generate/timediffusion_0.99'
log_file_path = os.path.join(output_directory, 'program_log.txt')

# 配置日志
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建输出目录
os.makedirs(output_directory, exist_ok=True)

# 超参数配置
batch_size = 2
epochs = 100
proximity = 0.99
samples = 6
step_granulation = 100

# 读取数据
history = pd.read_csv(file_path)
num_rows, num_columns = history.shape

# 设生成数据存储列表
synthetic_arrays = []

# 初始化训练集和测试集的累计时间
total_train_elapsed_time = 0
total_test_elapsed_time = 0

# 记录整体程序的开始时间
total_start_time = time.time()

# 生成数据（遍历原始数据并逐条训练生成）
for row in history.itertuples(index=False):
    seq = np.array(row).reshape(1, -1)

    # 创建TD模型对象，并传入输入形状信息
    model = TD(input_dims=seq.shape).to(device=device, dtype=torch.float32)

    # 记录模型训练的开始时间
    train_start_time = time.time()
    losses = model.fit(seq, epochs=epochs, batch_size=batch_size, verbose=True)
    train_end_time = time.time()

    # 记录模型测试的开始时间
    test_start_time = time.time()
    synthetic = model.synth(proximity=proximity, samples=samples, batch_size=batch_size,
                            step_granulation=step_granulation,
                            verbose=True)
    test_end_time = time.time()

    synthetic_numpy = synthetic.numpy().reshape((samples, num_columns))
    synthetic_arrays.append(synthetic_numpy)

    train_elapsed_time = train_end_time - train_start_time
    total_train_elapsed_time += train_elapsed_time

    test_elapsed_time = test_end_time - test_start_time
    total_test_elapsed_time += test_elapsed_time

# 使用vstack函数垂直堆叠数组（所有的生成数据）
synthetic_merged = np.vstack(synthetic_arrays)

# 将NumPy数组转换为DataFrame
synthetic_df = pd.DataFrame(synthetic_merged)

# 给最终生成的df结构的数据添加标签
column_names = [f'f{j + 1}' for j in range(num_columns)]
synthetic_df.columns = column_names

# 保存生成的数据到data文件夹
output_file_path = os.path.join(output_directory, 'not_ok_data_syn.csv')
logging.info(f'Output file path: {output_file_path}')
synthetic_df = synthetic_df.round(1)
# 将舍入后的列转换回整数类型
synthetic_df = synthetic_df.astype(int)
synthetic_df.to_csv(output_file_path, index=False)

# 记录整体程序的结束时间
total_end_time = time.time()

total_elapsed_time = total_end_time - total_start_time

# 获取当前时间的字符串表示
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 在每条日志记录之前添加包含当前时间的记录
with open(log_file_path, 'a') as log_file:
    log_file.write(f'当前时间: {current_time}\n')
    log_file.write(f'使用设备: {device}\n')
    log_file.write(f'存储路径: {output_file_path}\n')

    log_file.write(f"训练用时: {total_train_elapsed_time:.2f} 秒\n")
    log_file.write(f"测试用时: {total_test_elapsed_time:.2f} 秒\n")
    log_file.write(f"整体程序用时: {total_elapsed_time:.2f} 秒\n")
