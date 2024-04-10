"""
The corresponding labels for each query are processed to ensure that each query contains a label and that the ratio of img modality: text modality is roughly 1:1
"""

import argparse
import io
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import random
from tqdm import tqdm
random.seed(2023)
def get_one_doc_example(doc_path,output_file_path):

    data=[]
    input_data = pd.read_parquet(doc_path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()
        random_probability = random.random()

        # Retain the img label with a probability of 1/2
        if random_probability < 0.5:
            item_to_remove=['surround','text_id']
        else:
            item_to_remove=['alt','BUFFER','img_id']
        for item in item_to_remove:
            example.pop(item)
        data.append(example)
    # For removed item_key, it's value is set to nan(None)
    filter_data = pd.DataFrame(data)
    table = pa.Table.from_pandas(filter_data)
    pq.write_table(table, output_file_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--doc_path", type=str, default='clueweb_data/clueweb_data.parquet',)
    parser.add_argument("--output_file_path", type=str, default='clueweb_data/clueweb_data_one_label.parquet',
                        help="Path to output data file")
    args = parser.parse_args()
    get_one_doc_example(args.doc_path, args.output_file_path)