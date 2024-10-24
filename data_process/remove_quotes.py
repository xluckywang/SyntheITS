# -*- coding: utf-8 -*-

import csv


def remove_quotes(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_out:

        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        for row in reader:
            # 将字符串拆分成一个值的列表
            values = row[0].split(',')
            # 去除每个值的双引号
            cleaned_values = [value.replace('"', '') for value in values]
            # 将清理后的值写入CSV文件
            writer.writerow(cleaned_values)


# 用法示例
input_file_path = 'data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_train.csv'
output_file_path = 'data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_train.csv'
remove_quotes(input_file_path, output_file_path)
