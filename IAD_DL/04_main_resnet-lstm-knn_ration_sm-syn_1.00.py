import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import GlobalAveragePooling1D, Reshape, Conv1D, BatchNormalization, PReLU, Add, Bidirectional, LSTM, \
    GRU, Dense, Dropout, MultiHeadAttention, Input
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils.utils import read_data_syn, get_metrics, split_data_into_folds


# 定义构建残差块的函数
def residual_block(x, kernel_size, filters):
    # 第一个卷积层
    conv_x = Conv1D(filters, kernel_size, padding='same')(x)
    conv_x = BatchNormalization()(conv_x)  # 批量归一化
    conv_x = PReLU(shared_axes=[1])(conv_x)  # Parametric ReLU 激活函数

    # 第二个卷积层
    conv_y = Conv1D(filters, kernel_size, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)  # 批量归一化
    conv_y = PReLU(shared_axes=[1])(conv_y)  # Parametric ReLU 激活函数

    # 残差连接
    x = Add()([x, conv_y])
    x = BatchNormalization()(x)  # 批量归一化
    x = PReLU(shared_axes=[1])(x)  # Parametric ReLU 激活函数
    return x


# 定义构建 ResNet-1D 模型的函数
def build_resnet1d_model(input_shape, num_res_blocks=2, filters=32, kernel_size=5, lstm_units=32, num_heads=4):
    inputs = Input(shape=input_shape)  # 输入层

    # 第一个卷积层
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)  # 批量归一化
    x = PReLU(shared_axes=[1])(x)  # Parametric ReLU 激活函数

    res_x = x
    # 构建多个残差块
    for _ in range(num_res_blocks):
        res_x = residual_block(res_x, kernel_size, filters)

    # LSTM 层
    x = LSTM(lstm_units, dropout=0.2, return_sequences=True)(res_x)

    # 添加残差连接
    x = Add()([res_x, x])
    x = BatchNormalization()(x)  # 批量归一化
    x = PReLU(shared_axes=[1])(x)  # Parametric ReLU 激活函数

    # 对 LSTM 输出应用 MultiHeadAttention
    multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units // num_heads)(x, x)

    # 添加残差连接
    x = Add()([x, multi_head_attention])
    x = BatchNormalization()(x)  # 批量归一化
    x = PReLU(shared_axes=[1])(x)  # Parametric ReLU 激活函数

    # 重塑输出为 (None, lstm_units * 2)
    x = GlobalAveragePooling1D()(x)

    # 输出层，采用 sigmoid 激活函数
    outputs = Dense(1, activation='sigmoid')(x)

    # 构建模型
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Function to plot and save loss curve
def plot_loss_curve(train_loss, val_loss, save_path='loss_plot.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.clf()


# Function to build file paths
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


# Main function
def main():
    DATA_TYPE = 'sony_ai_robot_data'
    MODEL_TYPE = "timegan"
    RATIOS = [1.00]

    for RATIO in RATIOS:
        if DATA_TYPE == 'tightening_data':
            save_path = f"experimental_results_405/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/model_reslx"
            base_path_ori = os.path.join("data_sets_12_24", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", DATA_TYPE, "generate", MODEL_TYPE)
        else:
            save_path = f"experimental_results_405/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/model_reslx"
            base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

        file_paths = build_file_paths(base_path_ori, base_path_syn)

        X_train, X_test, Y_train_, Y_test_ = read_data_syn(
            file_paths['nor_syn'], file_paths['abn_syn'],
            file_paths['x_train'], file_paths['y_train'],
            file_paths['x_test'], file_paths['y_test'], ratio=RATIO
        )

        Y_train = np.argmax(Y_train_, axis=1)
        Y_test = np.argmax(Y_test_, axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        input_shape = X_train.shape[1:]

        if DATA_TYPE in ['tightening_data', 'sony_ai_robot_data']:
            KERNEL_SIZE = 5
        else:
            KERNEL_SIZE = 3
            BATCH_SIZE = 16

        model = build_resnet1d_model(input_shape, kernel_size=KERNEL_SIZE)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)
        history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_test, Y_test),
                            callbacks=[reduce_lr])

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plot_loss_curve(train_loss, val_loss, save_path=f"{save_path}/loss.png")

        lstm_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        X_train_features = lstm_feature_extractor.predict(X_train)
        X_test_features = lstm_feature_extractor.predict(X_test)

        knn_model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        }

        grid_search = GridSearchCV(knn_model, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_train_features, Y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Cross Validation Mean Score:", grid_search.best_score_)

        knn_model = grid_search.best_estimator_
        Y_pred = knn_model.predict(X_test_features)
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy:", accuracy)

        columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']

        y_pred_train = knn_model.predict_proba(X_train_features)
        evaluation_results_train = get_metrics(y_pred_train, Y_train_)
        df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
        df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)
        print(df_train_metrics)

        X_test_features_ = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[1], 1)
        X_test_folds, Y_test_folds = split_data_into_folds(X_test_features_, Y_test_, num_folds=5)

        metrics_list_all = [get_metrics(knn_model.predict_proba(np.squeeze(X_test_folds[i], axis=-1)), Y_test_folds[i])
                            for i in range(len(X_test_folds))]

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
