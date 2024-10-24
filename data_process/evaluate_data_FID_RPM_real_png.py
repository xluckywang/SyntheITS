from keras.layers import GlobalAveragePooling2D
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
import os

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


def generate_rpm_image(row_datas):
    """
    生成RPM图像

    参数:
    - speed: 速度数据，numpy数组

    返回值:
    - 无，生成并显示RPM图像
    """
    # z-标准化
    X = row_datas
    mu = np.mean(X)
    sigma = np.sqrt(np.var(X))
    Z = (X - mu) / sigma

    # PAA
    k = 2
    N = len(X)
    m = np.ceil(N / k).astype(int)
    X2 = np.zeros(m)

    if np.ceil(N / k) - np.floor(N / k) == 0:
        for i in range(m):
            X2[i] = 1 / k * np.sum(Z[k * (i - 1) + 1:k * i])
    else:
        for i in range(m - 1):
            X2[i] = 1 / k * np.sum(Z[k * (i - 1) + 1:k * i])
        X2[m - 1] = 1 / (N - k * (m - 1)) * np.sum(Z[k * (m - 1):])

    # 生成RPM图像
    M = np.tile(X2, (m, 1)) - np.tile(X2.reshape(-1, 1), (1, m))

    F = (M - np.min(M)) / (np.max(M) - np.min(M)) * 255

    return F


# 绘制相对位置矩阵图
def plot_and_get_curves(data, save_dir, color='green'):
    """
    绘制每行数据作为相对位置矩阵图像并返回图像数组列表

    Parameters:
        data (pandas.DataFrame): 数据DataFrame
        color (str): 曲线颜色，默认为'green'

    Returns:
        list of numpy.ndarray: 图像数组列表
    """
    image_arrays = []
    for index, row in data.iterrows():
        row_data = row.values  # 获取当前行的值并转换为数组
        if isinstance(row_data, (int, float)):
            # 如果 row_data 是单个数值，则将其转换为包含一个数值的数组
            row_data = np.array([row_data])
        elif not isinstance(row_data, np.ndarray):
            # 如果 row_data 既不是数值也不是数组，则跳过本次循环
            continue

        relative_position_matrix = generate_rpm_image(row_data)

        # 使用相对位置矩阵绘制图像
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.imshow(relative_position_matrix, cmap='viridis')
        ax.axis('off')
        # plt.show()
        # plt.close(fig)

        # 保存图像
        image_filename = f"{index}.png"
        image_path = os.path.join(save_dir, image_filename)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)

        # 获取当前图形的图像数组
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer._renderer)

        # 将图像调整为目标形状(299, 299)
        image_pil = Image.fromarray(image_array)
        resized_image = image_pil.resize((299, 299))

        # 将Image对象转换回数组
        result_array = np.array(resized_image)

        # 保留前三个通道，去除透明度通道
        result_array = result_array[:, :, :3]

        image_arrays.append(result_array)

        # 关闭图形以释放资源
        plt.close(fig)

    return image_arrays


if __name__ == '__main__':
    data_ype = "chinatown_data"  # chinatown_data     sony_ai_robot_data
    model_type = "timegan"  # timegan    timediffusion    doppelganger
    data_class = "not_ok"  # not_ok
    fig_type = "RPM"

    # 1、拧紧数据
    if data_ype == "tightening_data":
        real_data = pd.read_csv(
            f'data_sets_12_24_/tightening_data/original/{data_class}_data.csv')
        generated_data = pd.read_csv(
            f'data_sets_12_24_/tightening_data/generate/{model_type}/{data_class}_data_syn.csv')
        save_dir_real = f'data_sets_12_24_/tightening_data/original/{data_class}/{fig_type}'  # 修改为实际的文件夹路径
        save_dir_generated = f'data_sets_12_24_/tightening_data/generate/{model_type}/{data_class}/{fig_type}'  # 修改为实际的文件夹路径

    # 2、验证数据
    else:
        real_data = pd.read_csv(
            f'data_sets_12_24_/validation_data/{data_ype}/original/{data_class}_data.csv')
        generated_data = pd.read_csv(
            f'data_sets_12_24_/validation_data/{data_ype}/generate/{model_type}/{data_class}_data_syn.csv')
        save_dir_real = f'data_sets_12_24_/validation_data/{data_ype}/original/{data_class}/{fig_type}'  # 修改为实际的文件夹路径
        save_dir_generated = f'data_sets_12_24_/validation_data/{data_ype}/generate/{model_type}/{data_class}/{fig_type}'  # 修改为实际的文件夹路径

    # 调用新的绘制方法，获取图像数组
    real_images_list = plot_and_get_curves(real_data, save_dir_real)
    generated_images_list = plot_and_get_curves(generated_data, save_dir_generated)

    # # 将图像数组列表转换为 NumPy 数组
    # real_images = np.stack(real_images_list)
    # generated_images = np.stack(generated_images_list)
    #
    # # 显示图像数组的形状
    # print(real_images.shape)
    # print(generated_images.shape)
    #
    # # 计算FID得分
    # fid_score = calculate_fid_score(real_images, generated_images)
    #
    # print(f"FID Score: {fid_score}")

