import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from scipy.linalg import sqrtm
import numpy as np


# 计算FID得分
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


# 生成一些虚构图像和真实图像的示例数据
# 这里假设你有真实图像和生成图像的numpy数组，每个图像的大小为(299, 299, 3)
real_images = np.random.rand(10, 299, 299, 3)
generated_images = np.random.rand(100, 299, 299, 3)

# 计算FID得分
fid_score = calculate_fid_score(real_images, generated_images)

print(f"FID Score: {fid_score}")
