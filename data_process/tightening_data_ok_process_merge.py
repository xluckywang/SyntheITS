import pandas as pd
import glob
import chardet
from pathlib import Path

# 指定目标目录
target_directory = Path("tightening_data_all")

# 使用 Path 类构建匹配模式，查找所有 Ch* 文件
pattern = target_directory / "Ch*"

# 使用 glob 模块获取匹配的文件路径列表
h_files = glob.glob(str(pattern))

# 创建一个空的 DataFrame 来存储合并后的数据
combined_data = pd.DataFrame()

for h_file_01 in h_files:
    target_directory_01 = Path(h_file_01) / "P_*"
    h_files_01 = glob.glob(str(target_directory_01))

    for h_file_02 in h_files_01:
        target_directory_02 = Path(h_file_02) / "C_确认*.csv"
        h_files_02 = glob.glob(str(target_directory_02))

        for h_file_03 in h_files_02:
            target_directory_csv = Path(h_file_03)

            # 使用 chardet 检测文件编码
            with open(target_directory_csv, 'rb') as f:
                result = chardet.detect(f.read())
                encoding = result['encoding']

            # 检查文件是否存在且不为空
            if target_directory_csv.is_file() and target_directory_csv.stat().st_size > 0:
                try:
                    ero_rows = None  # 初始化变量
                    with open(target_directory_csv, 'r', encoding=encoding) as file:
                        for i, line in enumerate(file):
                            if line == '旋转角,旋转扭矩\n':
                                ero_rows = i
                                break

                    if ero_rows is not None:
                        print(f"在 {target_directory_csv} 中找到匹配的行在第 {ero_rows} 行")
                    else:
                        print(f"在 {target_directory_csv} 中未找到匹配的行")

                    # 读取文件时跳过表头，手动指定列名
                    start_row = ero_rows + 1 if ero_rows is not None else 0  # 如果 ero_rows 为 None，从第一行开始读取
                    df = pd.read_csv(target_directory_csv, header=0, names=["angle", "torque"], encoding=encoding,
                                     skiprows=range(start_row), usecols=[1])

                    column2_data = df.dropna()  # 过滤掉包含错误的行
                    # 获取第二列的所有数据
                    # column2_data = df["torque"]
                    # 将当前读取的数据按列依次合并到右侧
                    combined_data = pd.concat([combined_data, column2_data], axis=1)

                except pd.errors.ParserError as e:
                    print(f"Error reading {target_directory_csv}: {e}")
                    df_ = pd.DataFrame(columns=["angle", "torque"])
# 过滤掉包含异常值的列
combined_data = combined_data.dropna(axis=1)
combined_data = combined_data.transpose()
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)
# shuffled_data = shuffled_data.transpose()
num_columns = shuffled_data.shape[1]
column_names = ['f{}'.format(i + 1) for i in range(num_columns)]  # 创建列名列表
shuffled_data.columns = column_names  # 为DataFrame添加列名
sample_data = shuffled_data.head(300)

# 将打乱后的数据保存为一个新的 CSV 文件，并控制小数点后三位
shuffled_data_rounded = sample_data.round(3)  # 或者使用 shuffled_data.astype(float).round(3)
shuffled_data_rounded.to_csv(target_directory / 'tightening_data_ok_1.csv', index=False)

print("合并并随机打乱完成，并保存成功")
