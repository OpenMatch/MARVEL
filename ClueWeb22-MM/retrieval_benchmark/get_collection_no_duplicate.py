'''
Remove data that appears in the dataset and construct of the corpus
'''

import argparse
import io
import json
import os.path

from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

def remove_duplicate_dicts(input_list):
    # Used to store non-repeating dictionaries
    output_list = []

    # Used to record hashes that have already appeared
    seen_hashes = set()

    for d in input_list:
        # Convert the dictionary to a frozenset and then take its hash value
        d_hash = hash(frozenset(d.items()))
        # If this hash is not in a set where it has already appeared, it means unduplicated
        if d_hash not in seen_hashes:
            seen_hashes.add(d_hash)
            output_list.append(d)

    return output_list

def get_dump_example(path):
    text_dump = {}
    img_dump = {}
    input_data = pd.read_parquet(path)
    for index in tqdm(range(len(input_data))):
        example = input_data.iloc[index]
        example = example.to_dict()
        if example['text_id']!=None and example['img_id']!=None:
            raise ("Have some problems in data processing")
        if example['text_id'] != None:
            if example['surround'] not in text_dump:
                text_dump[example['surround']] = example['text_id']
        elif example['img_id'] != None:
            img=example['BUFFER']
            caption=example['alt']
            img_caption=f'{img}_{caption}'
            if img_caption not in img_dump:
                img_dump[img_caption] = example['img_id']
        else:
            raise ("there is no positive document")
    return text_dump,img_dump

#TODO:Since we have only 90,000 pieces of data after filtering, we start counting from 100,000 for documents that are not in the filtered dataset.
def duplicate_collection(collection_path,filter_img_dump,filter_doc_dump):
    img_dump=filter_img_dump
    text_dump=filter_doc_dump
    # Used to store data not used for train/dev/test datasets --> corpus
    text_data=[]
    img_data=[]
    img_count=100000
    text_count=100000
    input_data = pd.read_parquet(collection_path)
    for index in tqdm(range(len(input_data))):
        example = input_data.iloc[index]
        example = example.to_dict()
        if 'surround' in example:
            if example['surround'] not in text_dump:
                text_id='text_'+str(text_count)
                text_data.append({'text_id':text_id,'surround':example['surround']})
                text_dump[example['surround']] = text_id
                text_count += 1
            else:
                text_id=text_dump[example['surround']]
                text_data.append({'text_id': text_id, 'surround': example['surround']})
        if 'BUFFER' and 'alt' in example:
            img = example['BUFFER']
            caption = example['alt']
            img_caption = f'{img}_{caption}'
            if img_caption not in img_dump:
                img_id='img_'+str(img_count)
                img_data.append({'img_id':img_id,'caption':caption,'img':img})
                img_dump[img_caption] = img_id
                img_count+=1
            else:
                img_id=img_dump[img_caption]
                img_data.append({'img_id': img_id, 'caption': caption, 'img': img})

    text_data=remove_duplicate_dicts(text_data)
    img_data=remove_duplicate_dicts(img_data)
    print('-------the num of img_dump and img_data-------')
    print(len(img_dump),len(img_data))
    print('-------the num of text_dump and text_data--------')
    print(len(text_dump),len(text_data))
    return text_data, img_data






def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--collection_path", type=str, default='clueweb_data/raw_image_text_new.parquet', help="Path to data file")
    parser.add_argument("--filter_path", type=str, default='clueweb_data/clueweb_data_one_label.parquet', help="Path to data file")
    parser.add_argument('--output_path',type=str,default='clueweb22-mm/')
    args = parser.parse_args()

    filter_doc_dump,filter_img_dump=get_dump_example(args.filter_path)

    doc_data,img_data=duplicate_collection(args.collection_path,filter_img_dump,filter_doc_dump)
    doc_path=os.path.join(args.output_path,'text.parquet')
    img_path=os.path.join(args.output_path,'image.parquet')

    filter_doc_data = pd.DataFrame(doc_data)
    table_doc = pa.Table.from_pandas(filter_doc_data)
    pq.write_table(table_doc, doc_path)

    filter_img_data = pd.DataFrame(img_data)
    table_img = pa.Table.from_pandas(filter_img_data)
    pq.write_table(table_img, img_path)
    print(len(doc_data),len(img_data))

    print('-------finish-------')




if __name__ == '__main__':
    main()