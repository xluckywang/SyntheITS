import os
import pandas as pd
import tensorflow as tf
from classification_models.lstm import MyLSTM
from classification_models.lstm import train
from utils.utils import read_data, evaluate_model, split_data_into_folds
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, Flatten, Conv1D, BatchNormalization, Activation, Add
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 样本数据文件名称
TRAIN_DATA_PATH = "train_data.csv"
TRAIN_LABELS_PATH = "train_labels.csv"
TEST_DATA_PATH = "test_data.csv"
TEST_LABELS_PATH = "test_labels.csv"
OK_DATA_SYN_PATH = "ok_data_syn.csv"
NOT_OK_DATA_SYN_PATH = "not_ok_data_syn.csv"


def build_file_paths(base_path_ori, base_path_syn):
    return {
        'nor_syn': os.path.join(base_path_syn, OK_DATA_SYN_PATH),
        'abn_syn': os.path.join(base_path_syn, NOT_OK_DATA_SYN_PATH),
        'x_train': os.path.join(base_path_ori, TRAIN_DATA_PATH),
        'y_train': os.path.join(base_path_ori, TRAIN_LABELS_PATH),
        'x_test': os.path.join(base_path_ori, TEST_DATA_PATH),
        'y_test': os.path.join(base_path_ori, TEST_LABELS_PATH),
    }


def main():
    # 1. 定义训练数据名称、生成数据比例
    DATA_TYPE = 'sony_ai_robot_data'
    MODEL_TYPE = "timediffusion"
    # RATIO = 1.00
    RATIOS = [0.00, 0.25, 0.50, 0.75, 1.00]

    for RATIO in RATIOS:
        output_directory = f'experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/exp_output/'
        save_path = f"experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/model_resnet"

        # 2. 设置样本数据的基本路径、路径字典
        base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
        base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

        file_paths = build_file_paths(base_path_ori, base_path_syn)  # 路径字典

        # 3. 读取数据
        X_train, X_test, Y_train, Y_test = read_data(
            file_paths['nor_syn'], file_paths['abn_syn'],
            file_paths['x_train'], file_paths['y_train'],
            file_paths['x_test'], file_paths['y_test'], ratio=RATIO
        )
        Y_train = np.argmax(Y_train, axis=1)
        Y_test = np.argmax(Y_test, axis=1)

        # 对数据进行标准化处理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # 定义ResNet1D模型
        def resnet1d_block(inputs, filters, kernel_size, activation='relu'):
            x = Conv1D(filters, kernel_size, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            x = Conv1D(filters, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)

            # Residual connection
            x = Add()([x, inputs])
            x = Activation(activation)(x)
            return x

        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)

        x = Conv1D(64, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = resnet1d_block(x, 64, 3)
        x = resnet1d_block(x, 64, 3)

        x = LSTM(64)(x)

        # 定义模型的输出
        output = Dense(1, activation='sigmoid')(x)

        # 编译模型
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 训练模型
        model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

        # 提取训练集和测试集的特征表示
        lstm_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        X_train_features = lstm_feature_extractor.predict(X_train)
        X_test_features = lstm_feature_extractor.predict(X_test)

        # 将特征表示作为输入，构建XGBoost模型
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train_features, Y_train)

        # 在测试集上进行预测
        Y_pred = xgb_model.predict(X_test_features)

        # 计算准确性
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
