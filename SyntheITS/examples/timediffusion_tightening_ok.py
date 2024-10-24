import os
import time
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import torch
from src.synthesizers.timediffusion import TD

"""
版本02：
曲线单条分工艺段输入
"""
# 全局常量定义
TOLERANCE = 1e-10
TRAINING_EPOCHS = 100
BATCH_SIZE = 4


def process_curve(torque_data):
    # 计算一阶差分
    diff_torque = np.diff(torque_data)
    # diff_torque_abs = np.abs(diff_torque)
    # 找到可异分离点的索引(一阶差分的绝对值中最大的点）
    min_value = np.round(diff_torque.min(), decimals=2)
    # Find separation points with a tolerance for floating point comparison
    tolerance = 1e-10
    separation_min_points = np.where(np.isclose(diff_torque, min_value, rtol=tolerance))

    # min_row_indices = separation_min_points[0]
    min_column_indices = separation_min_points[1]

    max_before_min = np.max(diff_torque[:, :min_column_indices[0]])
    separation_max_points = np.where(np.isclose(diff_torque, max_before_min, rtol=tolerance))

    # max_row_indices = separation_max_points[0]
    max_column_indices = separation_max_points[1]

    # 输出可异分离点的索引
    print("快速拧紧阶段与慢速终紧阶段的分割点:", max_column_indices)
    print("慢速终紧阶段与拧紧结束阶段的分割点:", min_column_indices)

    # 分割时间序列数据：screw_in_phase、tightening_phase和finish_phase
    screw_in_phase = seq[:, :max_column_indices[0]]
    tightening_phase = seq[:, max_column_indices[0]:min_column_indices[0]]
    finish_phase = seq[:, min_column_indices[0]:]

    # 确保每个阶段的形状正确
    print("Screw In Phase Shape:", screw_in_phase.shape)
    print("Tightening Phase Shape:", tightening_phase.shape)
    print("Finish Phase Shape:", finish_phase.shape)
    return screw_in_phase, tightening_phase, finish_phase


def train_and_generate_synthetic_data(model, input_data, proximity=0.95, samples=6, step_granulation=200, verbose=True):
    model.to(device=device, dtype=torch.float32)
    losses = model.fit(input_data, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=verbose)
    synthetic_data = model.synth(proximity=proximity, samples=samples, batch_size=BATCH_SIZE,
                                 step_granulation=step_granulation, verbose=verbose)
    reshaped_data = synthetic_data.reshape(samples, len(input_data[0]))
    return reshaped_data, losses


# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f'Device in use: {device}')

# 文件路径配置
file_path = "../data_sets_12_24/tightening_data/original/ok_data.csv"
output_directory = '../data_sets_12_24/tightening_data/generate/timediffusion_01'
log_file_path = os.path.join(output_directory, 'program_log.txt')

# 配置日志
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建输出目录
os.makedirs(output_directory, exist_ok=True)

# 创建 sequence data
history = pd.read_csv(file_path)
history = history.transpose()
num_rows, num_columns = history.shape
column_names = ['curve{}'.format(i + 1) for i in range(num_columns)]
history.columns = column_names

# 记录整体程序的开始时间
total_start_time = time.time()

# 设生成数据存储列表
synthetic_arrays = []

screw_in_phase_losses_01 = []
tightening_phase_losses_02 = []
finish_phase_losses_03 = []

for loop_count in range(1):  # num_columns
    curve_num = "curve" + str(loop_count + 1)
    seq = history[curve_num].values.reshape(1, -1)
    single_torque_data = np.array(seq)
    seq_screw_in_phase, seq_tightening_phase, seq_finish_phase = process_curve(single_torque_data)

    # 训练和生成合成数据
    model_screw_in_phase = TD(input_dims=seq_screw_in_phase.shape)
    data_array_screw_in_phase, losses_01 = train_and_generate_synthetic_data(model_screw_in_phase, seq_screw_in_phase)
    screw_in_phase_losses_01.append(losses_01)

    model_tightening_phase = TD(input_dims=seq_tightening_phase.shape)
    data_array_tightening_phase, losses_02 = train_and_generate_synthetic_data(model_tightening_phase,
                                                                               seq_tightening_phase)
    tightening_phase_losses_02.append(losses_02)

    model_finish_phase = TD(input_dims=seq_finish_phase.shape)
    data_array_finish_phase, losses_03 = train_and_generate_synthetic_data(model_finish_phase, seq_finish_phase)
    finish_phase_losses_03.append(losses_03)

    # 转换成 DataFrame 结构
    data_array_screw_in_phase_numpy = data_array_screw_in_phase.cpu().numpy()
    data_array_tightening_phase_numpy = data_array_tightening_phase.cpu().numpy()
    data_array_finish_phase_numpy = data_array_finish_phase.cpu().numpy()

    # 将分阶段生成的数据拼接
    result_array = np.concatenate(
        [data_array_screw_in_phase_numpy, data_array_tightening_phase_numpy, data_array_finish_phase_numpy], axis=1)

    synthetic_arrays.append(result_array)

# 记录整体程序的结束时间
total_end_time = time.time()

# 01使用vstack函数垂直堆叠数组（所有的损失数据）
screw_merged = np.vstack(screw_in_phase_losses_01)
tightening_merged = np.vstack(tightening_phase_losses_02)
finish_merged = np.vstack(finish_phase_losses_03)
# 将NumPy数组转换为DataFrame
screw_df = pd.DataFrame(screw_merged)
tightening_df = pd.DataFrame(tightening_merged)
finish_df = pd.DataFrame(finish_merged)
screw_df.to_csv(f'{output_directory}/screw_losses_csv', index=False)
tightening_df.to_csv(f'{output_directory}/tightening_losses_csv', index=False)
finish_df.to_csv(f'{output_directory}/finish_losses_csv', index=False)

# 02使用vstack函数垂直堆叠数组（所有的生成数据）
synthetic_merged = np.vstack(synthetic_arrays)

# 将NumPy数组转换为DataFrame
synthetic_df = pd.DataFrame(synthetic_merged)

# 给最终生成的df结构的数据添加标签
column_names = [f'f{j + 1}' for j in range(num_rows)]
synthetic_df.columns = column_names

# 保存生成的数据到data文件夹
output_file_path = os.path.join(output_directory, 'ok_data_syn.csv')
logging.info(f'Output file path: {output_file_path}')
synthetic_df = synthetic_df.round(1)
synthetic_df.to_csv(output_file_path, index=False)

total_elapsed_time = total_end_time - total_start_time

# 获取当前时间的字符串表示
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 在每条日志记录之前添加包含当前时间的记录
with open(log_file_path, 'a') as log_file:
    log_file.write(f'当前时间: {current_time}\n')
    log_file.write(f'使用设备: {device}\n')
    log_file.write(f'存储路径: {output_file_path}\n')

    log_file.write(f"整体程序用时: {total_elapsed_time:.2f} 秒\n")
