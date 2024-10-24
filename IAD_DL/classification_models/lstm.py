import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class MyLSTM(tf.keras.Model):
    def __init__(self, units, n_classes, dropout_rate=0.0, hidden_activation='relu', output_activation='softmax',
                 name='lstmNetwork',
                 **kwargs):
        """
            初始化 MyLSTM 模型。

            参数：
            - units: LSTM 单元的数量。
            - n_classes: 输出类别的数量。
            - dropout_rate: Dropout 比率。
            - hidden_activation: 隐藏层的激活函数。
            - output_activation: 输出层的激活函数。
            - name: 模型的名称。
            - **kwargs: 其他参数传递给父类构造函数。
        """
        super(MyLSTM, self).__init__(name=name, **kwargs)

        # 定义模型的层
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, name="lstm1",
                                          kernel_initializer='orthogonal')
        self.lstm2 = tf.keras.layers.LSTM(units, name="lstm2", kernel_initializer='orthogonal')
        self.hidden_activation = tf.keras.layers.Activation(hidden_activation)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)

    def call(self, inputs, training=False):
        """
        定义模型的前向传播。

        参数：
        - inputs: 输入数据。
        - training: 是否在训练模式下。

        返回：
        - 模型的输出。
        """
        # 定义模型前向传播流程
        x, _, _ = self.lstm1(inputs, training=training)
        x = self.hidden_activation(x)
        outputs = self.lstm2(x, training=training)
        outputs = self.batch_norm(outputs, training=training)
        return self.model_output(outputs)


#
# class MyLSTM(tf.keras.Model):
#     def __init__(self, units, n_classes, dropout_rate=0.0, hidden_activation='relu', output_activation='softmax',
#                  name='lstmNetwork',
#                  **kwargs):
#         """
#         初始化 MyLSTM 模型。
#
#         参数：
#         - units: LSTM 单元的数量。
#         - n_classes: 输出类别的数量。
#         - dropout_rate: Dropout 比率。
#         - hidden_activation: 隐藏层的激活函数。
#         - output_activation: 输出层的激活函数。
#         - name: 模型的名称。
#         - **kwargs: 其他参数传递给父类构造函数。
#         """
#         super(MyLSTM, self).__init__(name=name, **kwargs)
#
#         # 定义模型的层
#         self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, name="lstm1",
#                                           kernel_initializer='orthogonal')
#         self.lstm2 = tf.keras.layers.LSTM(units, name="lstm2", kernel_initializer='orthogonal')
#         self.batch_norm = tf.keras.layers.BatchNormalization()
#         self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)
#
#     def call(self, inputs, training=False):
#         """
#         定义模型的前向传播。
#
#         参数：
#         - inputs: 输入数据。
#         - training: 是否在训练模式下。
#
#         返回：
#         - 模型的输出。
#         """
#         # 定义模型前向传播流程
#         inputs, _, _ = self.lstm1(inputs, training=training)
#         outputs = self.lstm2(inputs, training=training)
#         outputs = self.batch_norm(outputs, training=training)
#         return self.model_output(outputs)


# class MyLSTM(tf.keras.Model):
#     def __init__(self, units, n_classes, dropout_rate=0.0, hidden_activation='relu', output_activation='softmax',
#                  name='lstmNetwork', **kwargs):
#         """
#                 初始化 MyLSTM 模型。
#
#                 参数：
#                 - units: LSTM 单元的数量。
#                 - n_classes: 输出类别的数量。
#                 - dropout_rate: Dropout 比率。
#                 - hidden_activation: 隐藏层的激活函数。
#                 - output_activation: 输出层的激活函数。
#                 - name: 模型的名称。
#                 - **kwargs: 其他参数传递给父类构造函数。
#                 """
#         super(MyLSTM, self).__init__(name=name, **kwargs)
#
#         # 定义模型的层
#         self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, name="lstm1")
#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.lstm2 = tf.keras.layers.LSTM(units, name="lstm2")
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
#         self.batch_norm = tf.keras.layers.BatchNormalization()
#         self.model_output = tf.keras.layers.Dense(units=n_classes, activation=output_activation)
#
#     def call(self, inputs, training=False):
#         """
#                 定义模型的前向传播。
#
#                 参数：
#                 - inputs: 输入数据。
#                 - training: 是否在训练模式下。
#
#                 返回：
#                 - 模型的输出。
#                 """
#         # 定义模型前向传播流程
#         inputs, _, _ = self.lstm1(inputs, training=training)
#         inputs = self.dropout1(inputs, training=training)
#         outputs = self.lstm2(inputs, training=training)
#         outputs = self.dropout2(outputs, training=training)
#         outputs = self.batch_norm(outputs, training=training)
#         return self.model_output(outputs)


# def train(model, x_train, y_train, loss_object, optimizer, batch_size=32, n_epochs=1000,
#           early_stopping_patience=50, lr_scheduler=None, lr_scheduler_args=None, output_directory=None):
#     save_path_png = output_directory + 'loss_plot.png'
#     best_epoch_train_loss = float('inf')
#     best_epoch = -1
#     patience_count = 0  # 用于早停的计数器
#
#     iterations = int(np.ceil(x_train.shape[0] / batch_size))
#     # 新增列表用于存储每个epoch的total_loss
#     losses = []
#     for e in range(n_epochs):
#         # x_train, y_train = shuffle(x_train, y_train, random_state=0)
#         loss_iteration = 0
#         total_loss = 0.0
#
#         # 使用enumerate更干净地迭代批次
#         for ibatch, (batch_x, batch_y) in enumerate(get_batches(x_train, y_train, batch_size)):
#             with tf.GradientTape() as tape:
#                 predictions = model(batch_x, training=True)
#                 loss = loss_object(batch_y, predictions)
#                 loss_iteration += loss.numpy()
#                 gradients = tape.gradient(loss, model.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#         total_loss = loss_iteration / iterations
#
#         # 将total_loss添加到列表中
#         losses.append(total_loss)
#
#         # 计算训练集上的准确性
#         train_accuracy = calculate_accuracy(model, x_train, y_train)
#
#         if total_loss <= best_epoch_train_loss:
#             best_epoch_train_loss = total_loss
#             best_epoch = e
#
#             patience_count = 0  # 发现新的最佳模型时重置耐心计数
#         else:
#             patience_count += 1
#
#         print(f"第 {e} 轮 - 损失 {total_loss:.4f} - 训练准确性 {train_accuracy:.4f}")
#         print("===============")
#
#         # 早停检查
#         if patience_count >= early_stopping_patience:
#             print(f"在 {early_stopping_patience} 轮没有改进后提前停止。")
#             break
#     # 保存模型
#     # 保存模型为单一文件
#     model.save(save_path_model, save_format='tf', overwrite=True)
#
#     # 将 Training completed 的打印语句移动到循环结束后
#     print(
#         f"训练完成。最佳模型在第 {best_epoch} 轮找到，训练损失为 {best_epoch_train_loss:.4f}")
#
#     # 保存损失图像到指定文件夹
#     if save_path_png:
#         plt.plot(losses, label='Training Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.savefig(save_path_png)
#         plt.close()  # 关闭图形，以防止图形同时保存和展示


def train(model, x_train, y_train, x_test, y_test, loss_object, optimizer, batch_size=32, n_epochs=1000,
          early_stopping_patience=50, lr_scheduler=None, lr_scheduler_args=None, output_directory=None):
    save_path_png = os.path.join(output_directory, 'loss_plot.png')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    best_epoch_train_loss = float('inf')
    best_epoch = -1
    patience_count = 0

    iterations = int(np.ceil(x_train.shape[0] / batch_size))
    losses = []
    for e in range(n_epochs):
        # x_train, y_train = shuffle(x_train, y_train, random_state=0)  # Consider using a random seed

        loss_iteration = 0
        total_loss = 0.0

        for ibatch, (batch_x, batch_y) in enumerate(get_batches(x_train, y_train, batch_size)):
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = loss_object(batch_y, predictions)
                loss_iteration += loss.numpy()
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss = loss_iteration / iterations
        losses.append(total_loss)

        train_accuracy = calculate_accuracy(model, x_train, y_train)
        test_accuracy = calculate_accuracy(model, x_test, y_test)

        if total_loss <= best_epoch_train_loss:
            best_epoch_train_loss = total_loss
            best_epoch = e
            patience_count = 0
            # 加载模型前手动设置编译信息
            model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
            model.save(os.path.join(output_directory, "best_model"))
        else:
            patience_count += 1

        print(f"第 {e} 轮 - 损失 {total_loss:.8f} - 训练集准确性 {train_accuracy:.8f} - 测试集准确性 {test_accuracy:.8f}")
        print("===============")

        if patience_count >= early_stopping_patience:
            print(f"在 {early_stopping_patience} 轮没有改进后提前停止。")
            break

    print(f"训练完成。最佳模型在第 {best_epoch} 轮找到，训练损失为 {best_epoch_train_loss:.4f}")

    if save_path_png:
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path_png)
        plt.close()


def get_batches(x, y, batch_size):
    for i in range(0, x.shape[0], batch_size):
        start_id = i
        end_id = min(i + batch_size, x.shape[0])
        yield x[start_id:end_id], y[start_id:end_id]


def calculate_accuracy(model, x, y):
    predictions = model(x, training=False)
    y_argmax = np.argmax(y, axis=1)
    pred_argmax = np.argmax(predictions, axis=1)
    return accuracy_score(y_argmax, pred_argmax)
