import os
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Attention, Add, Activation, Conv1D, BatchNormalization, \
    GlobalAveragePooling1D, Reshape
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.utils import read_data, get_metrics, split_data_into_folds


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


def build_resnet1d_model(input_shape, num_blocks=2, filters=64, kernel_size=3, lstm_units=64):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(num_blocks):
        x = residual_block(x, filters, kernel_size)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    attention = Attention()([x, x])
    pooled = GlobalAveragePooling1D()(attention)
    x = Reshape((2 * lstm_units,))(pooled)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def residual_block(x, filters, kernel_size, activation='relu'):
    x_res = Conv1D(filters, kernel_size, padding='same')(x)
    x_res = BatchNormalization()(x_res)
    x_res = Activation(activation)(x_res)

    x_res = Conv1D(filters, kernel_size, padding='same')(x_res)
    x_res = BatchNormalization()(x_res)

    x = Add()([x_res, x])
    x = Activation(activation)(x)
    x = BatchNormalization()(x)
    return x


def train_model(model, X_train, Y_train, X_test, Y_test):
    early_stopping = EarlyStopping(monitor='loss', patience=30, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001)

    history = model.fit(X_train, Y_train, epochs=300, batch_size=32,
                        validation_data=(X_test, Y_test), callbacks=[early_stopping, reduce_lr], verbose=1)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    return train_loss, val_loss


def main():
    # 定义数据路径和模型参数
    DATA_TYPE = 'sony_ai_robot_data'
    MODEL_TYPE = "TT"
    RATIOS = [0.00, 0.25, 0.50, 0.75, 1.00]

    for RATIO in RATIOS:
        if DATA_TYPE == 'tightening_data':
            save_path = f"experimental_results_329/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/model_reslx"
            base_path_ori = os.path.join("data_sets_12_24", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", DATA_TYPE, "generate", MODEL_TYPE)
        else:
            save_path = f"experimental_results_329/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_{RATIO:.2f}/model_reslx"
            base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
            base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

        file_paths = build_file_paths(base_path_ori, base_path_syn)

        X_train, X_test, Y_train_, Y_test_ = read_data(file_paths['nor_syn'], file_paths['abn_syn'],
                                                       file_paths['x_train'], file_paths['y_train'],
                                                       file_paths['x_test'], file_paths['y_test'], ratio=RATIO)
        Y_train = np.argmax(Y_train_, axis=1)
        Y_test = np.argmax(Y_test_, axis=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        input_shape = X_train.shape[1:]
        model = build_resnet1d_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        train_loss, val_loss = train_model(model, X_train, Y_train, X_test, Y_test)

        plot_loss_curve(train_loss, val_loss, save_path=f"{save_path}/loss.png")

        lstm_feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
        X_train_features = lstm_feature_extractor.predict(X_train)
        X_test_features = lstm_feature_extractor.predict(X_test)

        lda_model = LDA()
        param_grid = {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto', 0.1, 0.5],
            'priors': [None, [0.5, 0.5], [0.3, 0.7]],
            'n_components': [None, 1]
        }

        grid_search = GridSearchCV(lda_model, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(X_train_features, Y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Cross Validation Mean Score:", grid_search.best_score_)

        lda_model = grid_search.best_estimator_
        Y_pred = lda_model.predict(X_test_features)
        accuracy = accuracy_score(Y_test, Y_pred)
        print("Accuracy:", accuracy)

        columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']
        y_pred_train = lda_model.predict_proba(X_train_features)
        evaluation_results_train = get_metrics(y_pred_train, Y_train_)

        df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
        df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)

        X_test_features_ = X_test_features.reshape(X_test_features.shape[0], X_test_features.shape[1], 1)
        X_test_folds, Y_test_folds = split_data_into_folds(X_test_features_, Y_test_, num_folds=5)

        metrics_list_all = [get_metrics(lda_model.predict_proba(np.squeeze(X_test_folds[i], axis=-1)), Y_test_folds[i])
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
    # 样本数据文件名称
    TRAIN_DATA_PATH = "train_data.csv"
    TRAIN_LABELS_PATH = "train_labels.csv"
    TEST_DATA_PATH = "test_data.csv"
    TEST_LABELS_PATH = "test_labels.csv"
    OK_DATA_SYN_PATH = "ok_data_syn.csv"
    NOT_OK_DATA_SYN_PATH = "not_ok_data_syn.csv"

    main()
