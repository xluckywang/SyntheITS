import pandas as pd

# 读取 Excel 文件
xls_file_path = 'data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_train.xls'  # 替换为你的 Excel 文件路径
df = pd.read_excel(xls_file_path)

# 保存为 CSV 文件
csv_file_path = 'data_sets_12_24/validation_data/share_price_in_crease_data/SharePriceIncrease_train.csv'
df.to_csv(csv_file_path, index=False)
