"""
    Get the stock data from Yahoo finance data
    Data from the period 01 January 2017 - 24 January 2021
"""
import pandas as pd

from preprocessing.utils import real_data_loading


def transformations(path, seq_len: int):
    # 从指定路径加载股票数据
    stock_df = pd.read_csv(path)

    try:
        # 将'Date'列设置为索引，并按日期对DataFrame进行排序
        stock_df = stock_df.set_index('Date').sort_index()
    except:
        # 如果设置索引失败，保留原始DataFrame
        stock_df = stock_df

    # 在将数据用于合成器模型之前应用数据转换
    processed_data, scaler = real_data_loading(stock_df.values, seq_len=seq_len)

    return processed_data, scaler
