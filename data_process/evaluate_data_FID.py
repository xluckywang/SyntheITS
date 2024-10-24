import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from scipy.linalg import sqrtm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def calculate_fid_score(real_images, generated_images):
    # 加载InceptionV3模型
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # 预处理图像
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)

    # 获取Inception模型的输出
    real_activations = model.predict(real_images)
    generated_activations = model.predict(generated_images)

    # 计算均值和协方差
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # 修正协方差矩阵为对称矩阵
    sigma_real = np.asarray((sigma_real + sigma_real.T) / 2.0)
    sigma_generated = np.asarray((sigma_generated + sigma_generated.T) / 2.0)

    # 计算FID得分
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

    return fid_score


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # 修复输入不是矩阵的问题
    try:
        sigma1_sqrtm = sqrtm(sigma1)
        sigma2_sqrtm = sqrtm(sigma2)
    except ValueError:
        # 如果 sqrtm 失败，则尝试使用其他方式确保输入是矩阵
        sigma1_sqrtm = np.linalg.cholesky(sigma1)
        sigma2_sqrtm = np.linalg.cholesky(sigma2)

    covmean = sigma1_sqrtm.dot(sigma2_sqrtm)

    # 防止复数问题
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算Fréchet距离
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# 其余代码保持不变


def plot_and_save_curve(data, output_path, color='green'):
    """
    绘制每行数据作为曲线并保存为图像

    Parameters:
        data (numpy.ndarray): 数据数组
        output_path (str): 输出图像的路径
        color (str): 曲线颜色，默认为'green'
    """
    # 绘制每行数据作为曲线
    plt.plot(data.T, color=color)  # 使用.T转置数据以便每行代表一个曲线
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_path, format='png', transparent=False)  # 不保存透明度通道
    plt.close()  # 关闭当前图形，以免影响后续绘图


def convert_image_to_array(image_path, target_size=(299, 299)):
    """
    读取保存的图像并将其转换为数组

    Parameters:
        image_path (str): 图像文件路径
        target_size (tuple): 目标大小，默认为 (299, 299)

    Returns:
        numpy.ndarray: 图像数组
    """
    # 读取保存的图像并将其转换为数组
    image = Image.open(image_path).convert('RGB')  # 将RGBA转换为RGB
    image_resized = image.resize(target_size, Image.ANTIALIAS)  # 调整大小
    image_array = np.array(image_resized)

    # 调整图像数组的形状
    image_array = image_array[np.newaxis, :, :, :]

    return image_array


output_path_1 = 'data_sets_12_24/tightening_data/original/output_image.png'
output_path_2 = 'data_sets_12_24/tightening_data/generate/timegan/output_image.png'
# 读取CSV文件
real_data = pd.read_csv('data_sets_12_24/tightening_data/original/ok_data.csv')
generated_data = pd.read_csv('data_sets_12_24/tightening_data/generate/timegan/ok_data_syn.csv')

# 调用绘制和保存方法
plot_and_save_curve(real_data, output_path_1, color='green')
plot_and_save_curve(generated_data, output_path_2, color='green')

# 调用转换方法
real_images = convert_image_to_array(output_path_1)
generated_images = convert_image_to_array(output_path_2)

# 显示图像数组的形状
print(real_images.shape)
print(generated_images.shape)

# 统一曲线颜色
line_color = 'green'  # 您可以替换为所需的颜色

# 计算FID得分
fid_score = calculate_fid_score(real_images, generated_images)

print(f"FID Score: {fid_score}")
