import os
import pandas as pd
import tensorflow as tf
from classification_models.lstm import MyLSTM
from classification_models.lstm import train
from utils.utils import read_data_syn, evaluate_model, split_data_into_folds
from datetime import datetime
import logging

# 超参数
BATCH_SIZE = 32
EPOCHS = 500
N_CLASSES = 2

LSTM_UNITS = 64
DROPOUT_RATE = 0.1
LEARNING_RATE = 0.001

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
    # start_time = datetime.now()  # 记录程序开始时间

    # 1. 定义训练数据名称、生成数据比例
    DATA_TYPE = 'sony_ai_robot_data'
    MODEL_TYPE = "TT"
    RATIO = 1.00

    # # 2. 设置日志格式
    # logging.basicConfig(
    #     filename=f'experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/log.log',
    #     level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #
    # logging.info("程序开始运行")

    # 2. 设置样本数据的基本路径、路径字典
    base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
    base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

    file_paths = build_file_paths(base_path_ori, base_path_syn)  # 路径字典

    save_path = f"experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/model_lstm"

    # 3. 读取数据
    X_train, X_test, Y_train, Y_test = read_data_syn(
        file_paths['nor_syn'], file_paths['abn_syn'],
        file_paths['x_train'], file_paths['y_train'],
        file_paths['x_test'], file_paths['y_test'], ratio=RATIO
    )

    # 3.1 将数据转换为 float32 类型
    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_test = Y_test.astype('float32')

    # 4. 实例化模型
    model = MyLSTM(LSTM_UNITS, N_CLASSES, dropout_rate=DROPOUT_RATE)

    # 5. 设置损失、优化器、训练模型
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train(model, X_train, Y_train, X_test, Y_test, loss_object, optimizer, batch_size=BATCH_SIZE, n_epochs=EPOCHS,
          early_stopping_patience=100, output_directory=f"{save_path}/")

    # 6. 设计相关指标列名、加载已经训练好的最好模型
    columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']
    # 加载模型
    loaded_model = tf.keras.models.load_model(os.path.join(save_path, "best_model"))

    # 7. 获取训练集评估数据、并保存
    evaluation_results_train = evaluate_model(loaded_model, X_train, Y_train)
    df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
    df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)
    print(df_train_metrics)

    # 8. 获取测试集评估数据（5等分测试集、收集每个测试集的评估指标、计算每个指标的均值和方差、保存数据）
    X_test_folds, Y_test_folds = split_data_into_folds(X_test, Y_test, num_folds=5)

    metrics_list_all = [evaluate_model(loaded_model, X_test_folds[i], Y_test_folds[i]) for i in
                        range(len(X_test_folds))]

    df_test = pd.DataFrame(metrics_list_all, columns=columns)
    mean_row = df_test.mean()
    std_row = df_test.std()

    df_test_metrics = pd.concat([df_test, mean_row.to_frame().T, std_row.to_frame().T], ignore_index=True)
    print(f'比例：{RATIO}================================================================================================')
    print(df_test_metrics)

    df_test_metrics.to_csv(f"{save_path}/test_result.csv", index=False)

    # end_time = datetime.now()  # 记录程序结束时间
    # logging.info(f"程序结束运行，结束时间：{end_time}")
    # elapsed_time = end_time - start_time
    # logging.info(f"程序运行时间：{elapsed_time}")
    # print(f"程序运行时间：{elapsed_time}")


if __name__ == "__main__":
    main()
