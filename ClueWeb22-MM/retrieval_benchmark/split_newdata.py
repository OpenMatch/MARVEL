# shuffled_data = input_data.sample(frac=1).reset_index(drop=True)
import argparse
import io
import json
import os.path

from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import random
random_seed=2023
random.seed(random_seed)
def get_train_dev_test(path):
    """
    Shuffle the data, then proceed to partition the dataset.

    """
    train_data=[]
    dev_data=[]
    test_data=[]
    input_data = pd.read_parquet(path)
    shuffled_data = input_data.sample(frac=1,random_state=random_seed).reset_index(drop=True)
    for index in tqdm(range(len(shuffled_data))):
        example = shuffled_data.iloc[index]
        example = example.to_dict()
        if index<10000:
            test_data.append(example)
        elif index>=10000 and index<20000:
            dev_data.append(example)
        else:
            train_data.append(example)
    print(len(train_data),len(dev_data),len(test_data))
    filter_train_data = pd.DataFrame(train_data)
    table_train = pa.Table.from_pandas(filter_train_data)
    filter_dev_data = pd.DataFrame(dev_data)
    table_dev = pa.Table.from_pandas(filter_dev_data)
    filter_test_data = pd.DataFrame(test_data)
    table_test = pa.Table.from_pandas(filter_test_data)
    return table_train,table_dev,table_test


if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--file_path", type=str, default='clueweb_data/clueweb_data_one_label_no_duplicate_id.parquet',)
    parser.add_argument("--out_path", type=str, default='clueweb22-mm/',
                        help="Path to output data file")
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    train_path=os.path.join(args.out_path,'train.parquet')
    dev_path=os.path.join(args.out_path,'dev.parquet')
    test_path=os.path.join(args.out_path,'test.parquet')
    train_data,dev_data,test_data=get_train_dev_test(args.file_path)

    pq.write_table(train_data, train_path)
    pq.write_table(dev_data, dev_path)
    pq.write_table(test_data, test_path)

    print('split datasets is finish!')