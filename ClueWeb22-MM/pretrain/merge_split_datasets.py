import argparse
import base64
import io
import os
import warnings

import clip
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='clueweb/filter2',
                        help="Path to data file")
    parser.add_argument("--output_path", type=str, default='pretrain/',
                        help="Path to data file")
    args = parser.parse_args()

    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]
    dfs = [pd.read_parquet(os.path.join(args.input_path, file)) for file in parquet_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    print(len(merged_df))


    dev_df = merged_df.sample(n=10000, random_state=1234).reset_index(drop=True)

    train_df = merged_df.drop(dev_df.index).reset_index(drop=True)
    print(len(dev_df))
    print(len(train_df))

    train_file_path = os.path.join(args.output_path, 'train.parquet')
    dev_file_path=os.path.join(args.output_path,'dev.parquet')
    dev_df.to_parquet(dev_file_path, index=False)
    train_df.to_parquet(train_file_path, index=False)
    print('finish')



if __name__ == '__main__':
    main()