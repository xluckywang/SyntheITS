import os
import pandas as pd
import tensorflow as tf
from classification_models.lstm import MyLSTM
from classification_models.lstm import train
from utils.utils import read_data, get_metrics, split_data_into_folds
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten, Conv1D, BatchNormalization, Activation, Add
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# 定义 EarlyStopping 回调 # val_loss
early_stopping = EarlyStopping(monitor='loss',  # 监控的指标是验证集的损失
                               patience=20,  # 如果在 10 个 epoch 内都没有改善就停止训练
                               verbose=1,  # 显示详细信息
                               restore_best_weights=True)  # 在停止时恢复最佳权重

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                 factor=0.5,
                                                 patience=10,
                                                 min_lr=0.0001)


def plot_loss_curve(train_loss, val_loss, save_path='loss_plot.png'):
    """
    绘制训练和验证损失曲线，并保存为图片文件。

    参数:
        train_loss: 训练集的损失列表
        val_loss: 验证集的损失列表
        save_path: 图片保存路径，默认为 'loss_plot.png'
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存图形为文件
    plt.savefig(save_path)

    # 清除图形
    plt.clf()


def build_file_paths(base_path_ori, base_path_syn):
    return {
        'nor_syn': os.path.join(base_path_syn, OK_DATA_SYN_PATH),
        'abn_syn': os.path.join(base_path_syn, NOT_OK_DATA_SYN_PATH),
        'x_train': os.path.join(base_path_ori, TRAIN_DATA_PATH),
        'y_train': os.path.join(base_path_ori, TRAIN_LABELS_PATH),
        'x_test': os.path.join(base_path_ori, TEST_DATA_PATH),
        'y_test': os.path.join(base_path_ori, TEST_LABELS_PATH),
    }


# 样本数据文件名称
TRAIN_DATA_PATH = "train_data.csv"
TRAIN_LABELS_PATH = "train_labels.csv"
TEST_DATA_PATH = "test_data.csv"
TEST_LABELS_PATH = "test_labels.csv"
OK_DATA_SYN_PATH = "ok_data_syn.csv"
NOT_OK_DATA_SYN_PATH = "not_ok_data_syn.csv"
n_feature_maps = 64


def main():
    # 1. 定义训练数据名称、生成数据比例
    DATA_TYPE = 'sony_ai_robot_data'
    MODEL_TYPE = "timediffusion"  # "timegan_and_timediffusion"
    RATIOS = [0.00, 0.25, 0.50, 0.75, 1.00]

    for RATIO in RATIOS:
        if DATA_TYPE == 'tightening_data':
            save_path = f"experimental_results_329/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/model_reslx"
            # 2. 设置样本数据的基本路径、路径字典、实验结果基本路径
            base_path_ori = os.path.join("data_sets_12_24", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", DATA_TYPE, "generate", MODEL_TYPE)

            file_paths = build_file_paths(base_path_ori, base_path_syn)  # 路径字典

        else:
            save_path = f"experimental_results_329/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/model_reslx"  # 实验结果基本路径
            # 2. 设置样本数据的基本路径、路径字典、实验结果基本路径
            base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

            file_paths = build_file_paths(base_path_ori, base_path_syn)  # 路径字典

        # 3. 读取数据
        X_train, X_test, Y_train_, Y_test_ = read_data(
            file_paths['nor_syn'], file_paths['abn_syn'],
            file_paths['x_train'], file_paths['y_train'],
            file_paths['x_test'], file_paths['y_test'], ratio=RATIO
        )
        Y_train = np.argmax(Y_train_, axis=1)
        Y_test = np.argmax(Y_test_, axis=1)

        # 对数据进行标准化处理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        def resnet1d_block(inputs, filters):
            conv_x = Conv1D(filters, kernel_size=8, padding='same')(inputs)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)

            conv_y = Conv1D(filters, kernel_size=5, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)

            conv_z = Conv1D(filters, kernel_size=3, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = Conv1D(filters, kernel_size=1, padding='same')(inputs)
            shortcut_y = BatchNormalization()(shortcut_y)

            output_block = Add()([shortcut_y, conv_z])
            output_block = Activation('relu')(output_block)
            return output_block

        def resnet1d_block_(inputs, filters):
            conv_x = Conv1D(filters, kernel_size=8, padding='same')(inputs)
            conv_x = BatchNormalization()(conv_x)
            conv_x = Activation('relu')(conv_x)

            conv_y = Conv1D(filters, kernel_size=5, padding='same')(conv_x)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)

            conv_z = Conv1D(filters, kernel_size=3, padding='same')(conv_y)
            conv_z = BatchNormalization()(conv_z)

            # no need to expand channels because they are equal
            shortcut_y = BatchNormalization()(inputs)

            output_block = Add()([shortcut_y, conv_z])
            output_block = Activation('relu')(output_block)

            return output_block

        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)

        # x = Conv1D(n_feature_maps, 3, padding='same')(inputs)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        output_block_1 = resnet1d_block(inputs, n_feature_maps)
        output_block_2 = resnet1d_block(output_block_1, n_feature_maps * 2)
        output_block_3 = resnet1d_block(output_block_2, n_feature_maps * 2)

        x = LSTM(n_feature_maps)(output_block_3)

        # # 添加新的维度，将2D张量转换为3D张量
        # x = tf.expand_dims(x, axis=2)

        # gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

        # 定义模型的输出
        output = Dense(1, activation='sigmoid')(x)

        # 编译模型
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 训练模型
        history = model.fit(X_train, Y_train,
                            epochs=1000,
                            batch_size=32,
                            validation_data=(X_test, Y_test),
                            callbacks=[early_stopping, reduce_lr])

        # 提取训练过程中的损失值和验证损失值
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # 调用方法绘制损失曲线，并保存到指定文件夹里
        plot_loss_curve(train_loss, val_loss, save_path=f"{save_path}/loss.png")

        # 提取训练集和测试集的特征表示
        lstm_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        X_train_features = lstm_feature_extractor.predict(X_train)
        X_test_features = lstm_feature_extractor.predict(X_test)

        # 将特征表示作为输入，构建XGBoost模型
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train_features, Y_train)

        # 在测试集上进行预测
        Y_pred = xgb_model.predict(X_test_features)
        # 在测试集上获取预测概率
        # y_pred_train = xgb_model.predict_proba(X_test_features)

        # 计算准确性
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy:", accuracy)

        # 6. 设计相关指标列名、保存地址
        columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']

        # 7. 获取训练集评估数据、并保存
        y_pred_train = xgb_model.predict_proba(X_train_features)

        evaluation_results_train = get_metrics(y_pred_train, Y_train_)

        df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
        df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)
        print(df_train_metrics)

        # 8. 获取测试集评估数据（5等分测试集、收集每个测试集的评估指标、计算每个指标的均值和方差、保存数据）
        X_test_features_ = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[1], 1)  # 增加一个维度
        X_test_folds, Y_test_folds = split_data_into_folds(X_test_features_, Y_test_, num_folds=5)  # 将测试数据分成5等分

        metrics_list_all = [
            get_metrics(xgb_model.predict_proba(np.squeeze(X_test_folds[i], axis=-1)), Y_test_folds[i]) for i
            in
            range(len(X_test_folds))]

        df_test = pd.DataFrame(metrics_list_all, columns=columns)
        mean_row = df_test.mean()
        std_row = df_test.std()

        df_test_metrics = pd.concat([df_test, mean_row.to_frame().T, std_row.to_frame().T], ignore_index=True)
        print(
            f'比例：{RATIO}================================================================================================')
        print(df_test_metrics)

        df_test_metrics.to_csv(f"{save_path}/test_result.csv", index=False)


if __name__ == "__main__":
    main()
