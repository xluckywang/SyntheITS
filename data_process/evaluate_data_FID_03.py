from keras.layers import GlobalAveragePooling2D
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from scipy.linalg import sqrtm
import numpy as np


# 计算FID得分
def calculate_fid_score(real_images, generated_images):
    # 加载修改后的InceptionV3模型
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling=None)
    x = GlobalAveragePooling2D()(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    # 预处理图像
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)

    # 获取Inception模型的输出
    real_activations = model.predict(real_images)
    generated_activations = model.predict(generated_images)

    # 计算均值和协方差
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # 计算FID得分
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

    return fid_score


# 计算Fréchet距离
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))

    # 防止复数问题
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算Fréchet距离
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def plot_and_get_curves(data, color='green'):
    """
    绘制每行数据作为曲线并返回图像数组列表

    Parameters:
        data (numpy.ndarray): 数据数组
        color (str): 曲线颜色，默认为'green'

    Returns:
        list of numpy.ndarray: 图像数组列表
    """
    image_arrays = []
    for row_data in data:
        # 绘制每行数据作为曲线
        plt.plot(data.iloc[:, row_data], color=color)
        plt.axis('off')  # 关闭坐标轴
        # plt.show()

        # 获取当前图形的图像数组
        fig = plt.gcf()
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)

        # 关闭当前图形，以免影响后续绘图
        plt.close()

        # 将图像调整为目标形状(299, 299)
        image_pil = Image.fromarray(image_array)
        resized_image = image_pil.resize((299, 299))

        # 将Image对象转换回数组
        result_array = np.array(resized_image)

        # 保留前三个通道，去除透明度通道
        result_array = result_array[:, :, :3]

        image_arrays.append(result_array)

    return image_arrays


# 读取CSV文件
real_data = pd.read_csv('data_sets_01_23/tightening_data/original/ok_data.csv')
generated_data = pd.read_csv('data_sets_01_23/tightening_data/generate/timegan/ok_data_syn.csv')

real_data = real_data.T
generated_data = generated_data.T

# 调用新的绘制方法，获取图像数组
real_images_list = plot_and_get_curves(real_data, color='green')
generated_images_list = plot_and_get_curves(generated_data, color='green')

# 将图像数组列表转换为 NumPy 数组
real_images = np.stack(real_images_list)
generated_images = np.stack(generated_images_list)

# 显示图像数组的形状
print(real_images.shape)
print(generated_images.shape)

# 计算FID得分
fid_score = calculate_fid_score(real_images, generated_images)

print(f"FID Score: {fid_score}")
