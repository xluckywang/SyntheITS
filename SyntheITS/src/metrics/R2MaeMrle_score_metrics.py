from keras import Input, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import GRU, Dense
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
import numpy as np
import pandas as pd


class StockPredictionModel:
    def __init__(self, units):
        """
        初始化股票预测模型。

        Args:
            units (int): RNN层的单元数。
        """
        self.units = units
        self.model = self.build_model()

    def build_model(self):
        """
        构建RNN模型。

        Returns:
            model: 构建好的RNN模型。
        """
        opt = Adam(name='AdamOpt')
        loss = MeanAbsoluteError(name='MAE')
        model = Sequential()
        model.add(GRU(units=self.units, name=f'RNN_1'))
        model.add(Dense(units=24, activation='sigmoid', name='OUT'))
        model.compile(optimizer=opt, loss=loss)
        return model

    def prepare_dataset(self, stock_data, synth_data, seq_len):
        """
        准备用于回归模型的数据集。

        Args:
            stock_data (numpy.ndarray): 真实股票数据。
            synth_data (numpy.ndarray): 合成的股票数据。
            seq_len (int): 序列长度。

        Returns:
            tuple: 包含训练和测试数据的元组。
        """
        stock_data = np.asarray(stock_data)
        synth_data = np.array(synth_data)[:len(stock_data)]
        n_events = len(stock_data)

        idx = np.arange(n_events)
        n_train = int(.75 * n_events)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_stock_train = stock_data[train_idx, :seq_len - 1, :]
        X_synth_train = synth_data[train_idx, :seq_len - 1, :]

        X_stock_test = stock_data[test_idx, :seq_len - 1, :]
        y_stock_test = stock_data[test_idx, -1, :]

        y_stock_train = stock_data[train_idx, -1, :]
        y_synth_train = synth_data[train_idx, -1, :]

        return X_stock_train, X_synth_train, X_stock_test, y_stock_test, y_stock_train, y_synth_train

    def train(self, X_stock_train, X_synth_train, X_stock_test, y_stock_train, y_synth_train, y_stock_test):
        """
        训练模型。

        Args:
            X_stock_train (numpy.ndarray): 真实股票数据的训练集。
            X_synth_train (numpy.ndarray): 合成股票数据的训练集。
            X_stock_test (numpy.ndarray): 真实股票数据的测试集。
            y_stock_train (numpy.ndarray): 真实股票数据的训练集标签。
            y_synth_train (numpy.ndarray): 合成股票数据的训练集标签。
            y_stock_test (numpy.ndarray): 真实股票数据的测试集标签。

        Returns:
            tuple: 包含真实数据训练结果和合成数据训练结果的元组。
        """
        early_stopping = EarlyStopping(monitor='val_loss')

        real_train = self._train_model(X_stock_train, y_stock_train, X_stock_test, y_stock_test, early_stopping)
        synth_train = self._train_model(X_synth_train, y_synth_train, X_stock_test, y_stock_test, early_stopping)

        return real_train, synth_train

    def _train_model(self, X_train, y_train, X_test, y_test, early_stopping):
        """
        训练模型的内部方法。

        Args:
            X_train (numpy.ndarray): 训练集。
            y_train (numpy.ndarray): 训练集标签。
            X_test (numpy.ndarray): 测试集。
            y_test (numpy.ndarray): 测试集标签。
            early_stopping (tensorflow.keras.callbacks.EarlyStopping): 提前停止训练的回调函数。

        Returns:
            history: 训练结果的历史记录。
        """
        history = self.model.fit(x=X_train,
                                 y=y_train,
                                 validation_data=(X_test, y_test),
                                 epochs=200,
                                 batch_size=128,
                                 callbacks=[early_stopping])
        return history


class MetricsEvaluator:
    def __init__(self, ts_real, ts_synth, X_stock_test, y_stock_test):
        self.ts_real = ts_real
        self.ts_synth = ts_synth
        self.X_stock_test = X_stock_test
        self.y_stock_test = y_stock_test

    def evaluate_metrics(self):
        real_predictions = self.ts_real.predict(self.X_stock_test)
        synth_predictions = self.ts_synth.predict(self.X_stock_test)

        metrics_dict = {'r2': [r2_score(self.y_stock_test, real_predictions),
                               r2_score(self.y_stock_test, synth_predictions)],
                        'MAE': [mean_absolute_error(self.y_stock_test, real_predictions),
                                mean_absolute_error(self.y_stock_test, synth_predictions)],
                        'MRLE': [mean_squared_log_error(self.y_stock_test, real_predictions),
                                 mean_squared_log_error(self.y_stock_test, synth_predictions)]}

        results = pd.DataFrame(metrics_dict, index=['Real', 'Synthetic'])
        return results
