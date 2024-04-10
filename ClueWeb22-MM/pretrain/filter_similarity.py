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
    parser.add_argument("--input_path", type=str, default='clueweb/filter1',
                        help="Path to data file")
    parser.add_argument("--output_path", type=str, default='clueweb/filter2',
                        help="Path to data file")
    parser.add_argument("--file_num", type=int, default=0,help="Path to data file")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Batch size to use')
    args = parser.parse_args()
    all_num_filtered=0
    all_num=0
    parquet_files = [file for file in os.listdir(args.input_path) if file.endswith('.parquet')]

    parquet_files.sort()
    for selected_file in parquet_files:

    # selected_file = parquet_files[args.file_num]

        input_file_path = os.path.join(args.input_path, selected_file)
        table = pq.read_table(input_file_path)
        df = table.to_pandas()
        filtered_df = df[df['similarity'] >= args.threshold]


        filtered_table = pa.Table.from_pandas(filtered_df)
        output_file_path = os.path.join(args.output_path, selected_file)
        # output_file_path = os.path.join(args.output_path, 'filter.parquet')
        pq.write_table(filtered_table, output_file_path)

        num_filtered = len(df) - len(filtered_df)
        print(f"Number of samples filtered out: {num_filtered}")
        print(f"Filtered data has been saved to {output_file_path}。")

        all_num_filtered += num_filtered
        all_num += len(filtered_df)


    print('finish')
    print(f"Total number of samples filtered out: {all_num_filtered}")
    print(f"Total number of samples retained: {all_num}")

if __name__ == '__main__':
    main()