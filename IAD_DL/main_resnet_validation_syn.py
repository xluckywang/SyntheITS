import os
import numpy as np
import pandas as pd
from classification_models.resnet import MyRESNET
from utils.utils import read_data_syn, get_metrics, split_data_into_folds

# x_val and y_val are only used to monitor the test loss and NOT for training
BATCH_SIZE = 32
EPOCHS = 1000
N_FEATURE_MAPS = 64

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
    MODEL_TYPE = "TT"
    RATIO = 1.00

    output_directory = f'experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/exp_output/'
    save_path = f"experimental_results/validation_data/{DATA_TYPE}/{MODEL_TYPE}/experiment_syn_{RATIO:.2f}/model_resnet"

    # 2. 设置样本数据的基本路径、路径字典
    base_path_ori = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "sample_separate")
    base_path_syn = os.path.join("data_sets_12_24", "validation_data", DATA_TYPE, "generate", MODEL_TYPE)

    file_paths = build_file_paths(base_path_ori, base_path_syn)  # 路径字典

    # 3. 读取数据
    X_train, X_test, Y_train, Y_test = read_data_syn(
        file_paths['nor_syn'], file_paths['abn_syn'],
        file_paths['x_train'], file_paths['y_train'],
        file_paths['x_test'], file_paths['y_test'], ratio=RATIO
    )

    # 4. 实例化模型
    Y_true = np.argmax(Y_test, axis=1)  # 获取测试集的真实标签
    nb_classes = len(np.unique(np.concatenate((Y_train, Y_test), axis=0)))  # 标签类别数量
    input_shape = X_train.shape[1:]  # 模型输入维度

    model = MyRESNET(output_directory, input_shape, nb_classes, N_FEATURE_MAPS, verbose=True)

    # 5. 训练模型
    model.fit(X_train, Y_train, X_test, Y_test, Y_true, BATCH_SIZE, EPOCHS)

    # 6. 设计相关指标列名、保存地址
    columns = ['accuracy', 'precision', 'recall', 'f1', 'app', 'anp']

    # 7. 获取训练集评估数据、并保存
    y_pred_train = model.predict(X_train, Y_train, return_df_metrics=False)

    evaluation_results_train = get_metrics(y_pred_train, Y_train)

    df_train_metrics = pd.DataFrame([evaluation_results_train], columns=columns)
    df_train_metrics.to_csv(f"{save_path}/train_result.csv", index=False)
    print(df_train_metrics)

    # 8. 获取测试集评估数据（5等分测试集、收集每个测试集的评估指标、计算每个指标的均值和方差、保存数据）
    X_test_folds, Y_test_folds = split_data_into_folds(X_test, Y_test, num_folds=5)  # 将测试数据分成5等分

    metrics_list_all = [
        get_metrics(model.predict(X_test_folds[i], Y_test_folds[i], return_df_metrics=False), Y_test_folds[i]) for i in
        range(len(X_test_folds))]

    df_test = pd.DataFrame(metrics_list_all, columns=columns)
    mean_row = df_test.mean()
    std_row = df_test.std()

    df_test_metrics = pd.concat([df_test, mean_row.to_frame().T, std_row.to_frame().T], ignore_index=True)
    print(f'比例：{RATIO}================================================================================================')
    print(df_test_metrics)

    df_test_metrics.to_csv(f"{save_path}/test_result.csv", index=False)


if __name__ == "__main__":
    main()
