import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Attention, Add, Activation, Conv1D, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.layers import GlobalAveragePooling1D

# 定义 EarlyStopping 回调 # val_loss
early_stopping = EarlyStopping(monitor='loss',  # 监控的指标是验证集的损失
                               patience=30,  # 如果在 10 个 epoch 内都没有改善就停止训练
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


def build_resnet_bidirectional_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    # 卷积层用于特征提取
    x = Conv1D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)

    # ResNet 块
    res = Add()([x, inputs])  # Residual connection
    res = Activation('relu')(res)

    # Bidirectional LSTM 层用于序列建模
    lstm = Bidirectional(LSTM(64, return_sequences=True))(res)

    # 注意力机制
    attention = Attention()([lstm, lstm])

    # 添加 GlobalAveragePooling1D 层，将时间序列数据降维为固定长度的向量
    pooled = GlobalAveragePooling1D()(attention)

    # 密集层，输出维度修改为 (None, 2)
    outputs = Dense(2, activation='softmax')(pooled)

    model = Model(inputs=inputs, outputs=outputs)
    return model


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
    MODEL_TYPE = "TT"  # "timegan_and_timediffusion"
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
        X_train, X_test, Y_train, Y_test = read_data(
            file_paths['nor_syn'], file_paths['abn_syn'],
            file_paths['x_train'], file_paths['y_train'],
            file_paths['x_test'], file_paths['y_test'], ratio=RATIO
        )
        # Y_train = np.argmax(Y_train_, axis=1)
        # Y_test = np.argmax(Y_test_, axis=1)

        # 对数据进行标准化处理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # 构建模型
        input_shape = X_train.shape[1:]
        model = build_resnet_bidirectional_lstm_attention_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # 模型编译时使用 sparse_categorical_crossentropy
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 训练模型
        history = model.fit(X_train, Y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(X_test, Y_test),
                            callbacks=[early_stopping, reduce_lr])

        # 模型预测
        Y_pred_prob = model.predict(X_test)
        Y_pred = np.argmax(Y_pred_prob, axis=1)

        # 绘制损失曲线
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plot_loss_curve(train_loss, val_loss, save_path='loss.png')

        # 提取特征表示
        lstm_feature_extractor = Model(inputs=model.input, outputs=model.layers[-4].output)
        X_train_features = lstm_feature_extractor.predict(X_train)
        X_test_features = lstm_feature_extractor.predict(X_test)

        # 构建LDA模型
        lda_model = LDA()

        # 将特征转换为二维数组
        X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
        X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

        # 更新参数网格
        param_grid = {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5],
            'priors': [None, [0.5, 0.5], [0.3, 0.7]],
            'n_components': [None, 1]  # 用于二分类问题
        }

        # 尝试排除特定参数组合
        excluded_param_combinations = [
            {'solver': 'lsqr', 'n_components': None},  # 排除这种组合
            # 可以继续添加其他需要排除的组合
        ]

        # 使用列表推导式生成新的参数组合
        new_param_grid = [param for param in param_grid if param not in excluded_param_combinations]

        # 输出新的参数网格，检查是否正确
        print("New Parameter Grid:")
        print(new_param_grid)

        # 更新 GridSearchCV
        grid_search = GridSearchCV(lda_model, param_grid, cv=10, scoring='accuracy')

        # lda_model.fit(X_train_features, Y_train)
        # 使用训练数据进行参数搜索
        grid_search.fit(X_train_features, Y_train)
        # 输出最优参数
        print("Best Parameters:", grid_search.best_params_)

        # 输出交叉验证的平均得分
        print("Cross Validation Mean Score:", grid_search.best_score_)

        # 使用最优参数的模型进行预测
        lda_model = grid_search.best_estimator_

        # 在测试集上进行预测
        Y_pred = lda_model.predict(X_test_features)

        # 计算准确性
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy:", accuracy)

        # 6. 设计相关指标列名、保存地址
        columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']

        # 7. 获取训练集评估数据、并保存
        y_pred_train = lda_model.predict_proba(X_train_features)

        evaluation_results_train = get_metrics(y_pred_train, Y_train)

        df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
        df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)
        print(df_train_metrics)

        # 8. 获取测试集评估数据（5等分测试集、收集每个测试集的评估指标、计算每个指标的均值和方差、保存数据）
        X_test_features_ = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[1], 1)  # 增加一个维度
        X_test_folds, Y_test_folds = split_data_into_folds(X_test_features_, Y_test, num_folds=5)  # 将测试数据分成5等分

        metrics_list_all = [
            get_metrics(lda_model.predict_proba(np.squeeze(X_test_folds[i], axis=-1)), Y_test_folds[i]) for i
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
