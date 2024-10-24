# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf, pacf
from scipy.linalg import norm
from scipy.fft import fft

# 设置随机种子
np.random.seed(42)

# 导入数据
original_path = 'data_sets_12_24/tightening_data/original/ok_data.csv'
generated_path = 'data_sets_12_24/tightening_data/generate/timediffusion/ok_data_syn.csv'

df_generated = pd.read_csv(generated_path)
df_original = pd.read_csv(original_path)

# 从 df_generated 中抽取与 df_original 具有相同样本数的样本
df_generated_subset = df_generated.sample(n=len(df_original), replace=True, random_state=42)

# 1. Mean Squared Error (MSE)
mse = mean_squared_error(df_original, df_generated_subset)
print(f'Mean Squared Error (MSE): {mse}')

# 2. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 3. Mean Absolute Error (MAE)
mae = mean_absolute_error(df_original, df_generated_subset)
print(f'Mean Absolute Error (MAE): {mae}')

# 4. Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((df_original - df_generated_subset) / df_original)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# # 5. Time Series Cross-Validation
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, test_index in tscv.split(df_original):
#     train, test = df_generated_subset.iloc[train_index], df_original.iloc[test_index]
#     # Perform model training and evaluation on the split
#
# # 6. Autocorrelation and Partial Autocorrelation Functions
# acf_values = acf(df_generated_subset.values.ravel(), nlags=23)
# print('Autocorrelation Function (ACF):', acf_values)
# pacf_values = pacf(df_generated_subset.values.ravel(), nlags=7)
# print('Partial Autocorrelation Function (PACF):', pacf_values)
#
# # 7. Spectral Distortion
# spectral_distortion = norm(fft(df_generated_subset, axis=1) - fft(df_original, axis=1))
# print(f'Spectral Distortion: {spectral_distortion}')
#
# # 8. Fourier Transform
# fft_generated = fft(df_generated_subset, axis=1)
# fft_original = fft(df_original, axis=1)
# Compare the frequency domain representations

# 9. Time Series Model Evaluation Metrics (e.g., AIC, BIC)
# Perform evaluation using appropriate time series models

# 10. Feature and Pattern Observations
# Examine whether generated data captures features and patterns of the original data
