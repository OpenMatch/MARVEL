"""
Merge the date for next processes
"""
import argparse
import json
import os.path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--input_root_path", type=str,
                        default='clueweb_data/clueweb_data')
    parser.add_argument("--output_root_path", type=str, default='/data3/meisen/clueweb-imgtext/thirdfilter')
    args = parser.parse_args()
    os.makedirs(args.output_root_path, exist_ok=True)
    parquet_files = [f for f in os.listdir(args.input_root_path) if f.endswith('.parquet')]
    dfs = []
    for file in parquet_files:
        file_path = os.path.join(args.input_root_path, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    sample_count = len(combined_df)
    print("Total number of samples:", sample_count)

    # Save the merged data to a new parquet file
    output_file_path = os.path.join(args.output_root_path, 'combined_data.parquet')
    combined_df.to_parquet(output_file_path, index=False)

    print('------------finish----------------')


if __name__ == '__main__':
    main()





