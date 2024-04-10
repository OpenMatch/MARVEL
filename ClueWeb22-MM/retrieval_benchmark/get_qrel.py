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
def get_qrels(example):
    qrel_list = []
    qid = example['qid']
    if example['img_id']!=None:
        img_id=example['img_id']
        qrel_list.append([qid, img_id, '1'])
    if example['text_id']!=None:
        did=example['text_id']
        qrel_list.append([qid, did, '1'])
    return qrel_list


def get_test_label(test_path):
    """
    get the label of the test dateset
    """
    test_qrels=[]
    count=0
    input_data = pd.read_parquet(test_path)
    for index in tqdm(range(len(input_data))):
        example = input_data.iloc[index]
        example = example.to_dict()
        if example['img_id']!=None and example['text_id']!=None:
            count+=1
        test_qrels.extend(get_qrels(example))
    print(count)
    print(len(test_qrels))
    return test_qrels

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--input_path", type=str, default='clueweb22-mm/test.parquet',)
    parser.add_argument("--output_path", type=str, default='clueweb22-mm/test_qrels.txt',
                        help="Path to output data file")
    args = parser.parse_args()

    test_qrels=get_test_label(args.input_path)
    with open(args.output_path, "w") as fout:
        for example in test_qrels:
            fout.write("\t".join(example) + "\n")
    print('----------finished----------')





