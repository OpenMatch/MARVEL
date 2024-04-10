import pandas as pd
import math

# 读取原始 Parquet 文件为 DataFrame
input_data = pd.read_parquet('clueweb22-mm/image.parquet')
input_data.drop(141666,inplace=True)
input_data.to_parquet('clueweb22-mm/image_new.parquet')