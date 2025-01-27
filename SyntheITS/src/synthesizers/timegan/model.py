"""
    TimeGAN class implemented accordingly with:
    Original code can be found here: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/master/alg/timegan/
"""
from tqdm import tqdm
import numpy as np
from pandas import DataFrame
from typing import Optional, List

from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor, float32
from tensorflow import data as tfdata
from tensorflow import nn
from keras import (Model, Sequential, Input)
from keras.layers import (GRU, LSTM, Dense)
from keras.optimizers import Adam
from keras.losses import (BinaryCrossentropy, MeanSquaredError)

from synthesizers.base import BaseGANModel, ModelParameters, TrainParameters
from preprocessing.utils import real_data_loading


def make_net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    # 构建生成器和判别器模型的中间网络层
    # net_type 可以选择使用 GRU 或 LSTM 单元
    if net_type == 'GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                          return_sequences=True,
                          name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                           return_sequences=True,
                           name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model


class TimeGAN(BaseGANModel):
    __MODEL__ = 'TimeGAN'

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)
        self.seq_len = None
        self.n_seq = None
        self.hidden_dim = model_parameters.latent_dim
        self.gamma = model_parameters.gamma
        self.num_cols = None

    def fit(self, data: DataFrame,
            train_arguments: TrainParameters,
            num_cols: Optional[List[str]] = None,
            cat_cols: Optional[List[str]] = None):
        """
        Fits the TimeGAN model.

        Args:
            data: A pandas DataFrame with the data to be synthesized.
            train_arguments: TimeGAN training arguments.
            num_cols: List of columns to be handled as numerical
            cat_cols: List of columns to be handled as categorical
        """
        super().fit(data=data, num_cols=num_cols, cat_cols=cat_cols, train_arguments=train_arguments)
        if cat_cols:
            raise NotImplementedError("TimeGAN does not support categorical features.")
        self.num_cols = num_cols
        self.seq_len = train_arguments.sequence_length
        self.n_seq = train_arguments.number_sequences
        # 处理输入数据
        processed_data, scaler = real_data_loading(data[self.num_cols].values, seq_len=self.seq_len)
        # 训练模型
        self.train(data=processed_data, train_steps=train_arguments.epochs)

    def sample(self, n_samples: int):
        """
        Samples new data from the TimeGAN.

        Args:
            n_samples: Number of samples to be generated.
        """
        Z_ = next(self.get_batch_noise(size=n_samples))  # 从噪声数据中获取一个批次的噪声数据，大小为n_samples
        # 使用生成器模型生成数据
        records = self.generator(Z_)
        data = []  # 存储生成的数据样本
        for i in range(records.shape[0]):  # 遍历生成的数据样本
            data.append(
                DataFrame(records[i], columns=self.num_cols))  # 将生成的样本数据转换为DataFrame格式，并将每个样本的列命名为self.num_cols中的列名
        return data  # 返回生成的数据样本

    def define_gan(self):
        # 定义 TimeGan 模型的各个组成部分

        # 定义生成器的辅助模型
        self.generator_aux = Generator(self.hidden_dim).build()
        # 定义监督器模型
        self.supervisor = Supervisor(self.hidden_dim).build()
        # 定义判别器模型
        self.discriminator = Discriminator(self.hidden_dim).build()
        # 定义恢复器模型
        self.recovery = Recovery(self.hidden_dim, self.n_seq).build()
        # 定义嵌入器模型
        self.embedder = Embedder(self.hidden_dim).build()

        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        # --------------------------------
        # Building the AutoEncoder
        # --------------------------------
        # 构建自编码器
        H = self.embedder(X)
        X_tilde = self.recovery(H)
        # 创建自编码器模型
        self.autoencoder = Model(inputs=X, outputs=X_tilde)

        # ---------------------------------
        # Adversarial Supervise Architecture
        # ---------------------------------
        # 构建监督器的生成器模型
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)
        # 创建监督器的生成器模型
        self.adversarial_supervised = Model(inputs=Z,
                                            outputs=Y_fake,
                                            name='AdversarialSupervised')

        # ---------------------------------
        # Adversarial architecture in latent space
        # ---------------------------------
        # 构建嵌入器的生成器模型
        Y_fake_e = self.discriminator(E_Hat)
        # 创建嵌入器的生成器模型
        self.adversarial_embedded = Model(inputs=Z,
                                          outputs=Y_fake_e,
                                          name='AdversarialEmbedded')
        # ---------------------------------
        # Synthetic data generation
        # ---------------------------------
        # 使用监督器生成合成数据
        X_hat = self.recovery(H_hat)
        # 创建生成器模型
        self.generator = Model(inputs=Z,
                               outputs=X_hat,
                               name='FinalGenerator')

        # --------------------------------
        # Final discriminator model
        # --------------------------------
        # 创建判别器模型
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name="RealDiscriminator")

        # ----------------------------
        # Define the loss functions
        # ----------------------------
        # 定义损失函数
        self._mse = MeanSquaredError()
        self._bce = BinaryCrossentropy()

    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @function
    def train_embedder(self, x, opt):
        with GradientTape() as tape:
            # Supervised Loss
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z):
        # Loss on false negatives
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        # Loss on false positives
        y_fake = self.adversarial_supervised(z)
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        y_fake_e = self.adversarial_embedded(z)
        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @function
    def train_generator(self, x, z, opt):
        with GradientTape() as tape:
            y_fake = self.adversarial_supervised(z)
            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            y_fake_e = self.adversarial_embedded(z)
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            x_hat = self.generator(z)
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @function
    def train_discriminator(self, x, z, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data)
                    .shuffle(buffer_size=n_windows)
                    .batch(self.batch_size).repeat())

    def _generate_noise(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self, size=None):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                    .batch(self.batch_size if size is None else size)
                    .repeat())

    def train(self, data, train_steps):
        # Assemble the model
        self.define_gan()  # 定义GAN模型

        ## Embedding network training
        autoencoder_opt = Adam(learning_rate=self.g_lr)  # 使用Adam优化器和给定的学习率
        for _ in tqdm(range(train_steps), desc='Emddeding network training'):  # 根据给定的训练步数进行循环训练
            X_ = next(self.get_batch_data(data, n_windows=len(data)))  # 从数据中获取一个批次的数据
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)  # 使用自动编码器模型训练该批次的数据，并记录训练的损失值

        ## Supervised Network training
        supervisor_opt = Adam(learning_rate=self.g_lr)  # 使用Adam优化器和给定的学习率
        for _ in tqdm(range(train_steps), desc='Supervised network training'):  # 根据给定的训练步数进行循环训练
            X_ = next(self.get_batch_data(data, n_windows=len(data)))  # 从数据中获取一个批次的数据
            step_g_loss_s = self.train_supervisor(X_, supervisor_opt)  # 使用监督网络模型训练该批次的数据，并记录训练的损失值

        ## Joint training
        generator_opt = Adam(learning_rate=self.g_lr)  # 使用Adam优化器和给定的学习率
        embedder_opt = Adam(learning_rate=self.g_lr)  # 使用Adam优化器和给定的学习率
        discriminator_opt = Adam(learning_rate=self.d_lr)  # 使用Adam优化器和给定的学习率

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0  # 初始化损失值为0
        for _ in tqdm(range(train_steps), desc='Joint networks training'):  # 根据给定的训练步数进行循环训练

            # Train the generator (k times as often as the discriminator)
            # Here k=2
            for _ in range(2):  # 生成器的训练次数为鉴别器训练次数的两倍
                X_ = next(self.get_batch_data(data, n_windows=len(data)))  # 从数据中获取一个批次的数据
                Z_ = next(self.get_batch_noise())  # 获取一个批次的噪声数据
                # --------------------------
                # Train the generator
                # --------------------------
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_,
                                                                                   generator_opt)  # 使用生成器模型训练该批次的数据，并记录训练的损失值

                # --------------------------
                # Train the embedder
                # --------------------------
                step_e_loss_t0 = self.train_embedder(X_, embedder_opt)  # 使用嵌入网络模型训练该批次的数据，并记录训练的损失值

            X_ = next(self.get_batch_data(data, n_windows=len(data)))  # 从数据中获取一个批次的数据
            Z_ = next(self.get_batch_noise())  # 获取一个批次的噪声数据
            step_d_loss = self.discriminator_loss(X_, Z_)  # 使用鉴别器网络模型计算鉴别器的损失值
            if step_d_loss > 0.15:  # 如果鉴别器的损失值大于0.15
                step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)  # 进行鉴别器网络模型的训练，并更新损失值


class Generator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self):
        model = Sequential(name='Generator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim,
                         net_type=self.net_type)
        return model


class Discriminator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self):
        model = Sequential(name='Discriminator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=1,
                         net_type=self.net_type)
        return model


class Recovery(Model):
    def __init__(self, hidden_dim, n_seq):
        self.hidden_dim = hidden_dim
        self.n_seq = n_seq
        return

    def build(self):
        recovery = Sequential(name='Recovery')
        recovery = make_net(recovery,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.n_seq)
        return recovery


class Embedder(Model):

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        return

    def build(self):
        embedder = Sequential(name='Embedder')
        embedder = make_net(embedder,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.hidden_dim)
        return embedder


class Supervisor(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def build(self):
        model = Sequential(name='Supervisor')
        model = make_net(model,
                         n_layers=2,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim)
        return model
