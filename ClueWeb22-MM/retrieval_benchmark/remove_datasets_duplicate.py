"""
Unifies the ids of the labels in the dataset, preventing the phenomenon of consistent content but different corresponding ids
"""
import argparse
import io
import json
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
def remove_duplicate_id(doc_path):

    data=[]
    text_dump={} #27325
    img_dump={} #28098
    input_data = pd.read_parquet(doc_path)
    for index in tqdm(range(len(input_data))):
        example = input_data.iloc[index]
        example = example.to_dict()
        if example['text_id'] !=None:
            if example['surround'] not in text_dump:
                text_dump[example['surround']]=example['text_id']
        elif example['img_id'] != None:
            img=example['BUFFER']
            caption=example['alt']
            img_caption=f'{img}_{caption}'
            if img_caption not in img_dump:
                img_dump[img_caption] = example['img_id']
        else:
            raise ("there is no positive document")
    for index in tqdm(range(len(input_data))):
        example = input_data.iloc[index]
        example = example.to_dict()
        tmp=example.copy()
        if example['text_id'] !=None:
            tmp['text_id']=text_dump[example['surround']]
        elif example['img_id'] != None:
            img=example['BUFFER']
            caption=example['alt']
            img_caption=f'{img}_{caption}'
            tmp['img_id'] = img_dump[img_caption]

        data.append(tmp)
    return data

if __name__=='__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--input_path", type=str, default='clueweb_data/clueweb_data_one_label.parquet',)
    parser.add_argument("--output_path", type=str, default='clueweb_data/clueweb_data_one_label_no_duplicate_id.parquet',
                        help="Path to output data file")
    args = parser.parse_args()
    # remove the duplicate id.
    filter_data=remove_duplicate_id(args.input_path)
    filter_data = pd.DataFrame(filter_data)
    table = pa.Table.from_pandas(filter_data)
    pq.write_table(table, args.output_path)
    print('the remove duplicate id is finish!')

