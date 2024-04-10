import argparse
import json

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
filter_out_keyword_set = {
    'website',
    'official website',
    'original',
    'view website',
    'visit website',
    'visit our website',
    'visit website',
    'visit site',
    'home',
    'home page',
    'homepage',
    'about',
    'about us',
    'here',
    'this',
    'this article',
    'this page',
    'click here',
    'link',
    'source link',
    'offsite link',
    'this link',
    'more',
    'more info',
    'more information',
    'view more',
    'learn more',
    'read more',
    'see more',
    'find out more',
    'english',
    'en',
    'download',
    'save',
    'login',
    'sign in',
    'sign up',
    'register',
    'reply',
}
filter_out_keyword_set = {x.replace(' ','') for x in filter_out_keyword_set}

def process_data(data):
    all_data = []
    for example in data:
        anchor_list = example['anchors']
        cw22id = example['ClueWeb22-ID']

        seen = set()
        unique_list = [sublist for sublist in anchor_list if sublist[2] not in seen and not seen.add(sublist[2])]
        for anchor in unique_list:
            query_text = anchor[2]
            if len(query_text.split()) > 64:
                continue
            if len(query_text.split()) < 5:
                continue
            if query_text.replace(' ', '').lower() in filter_out_keyword_set:
                continue
            all_data.append({'query': query_text, 'label': cw22id})
    return all_data

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--doc_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--qry_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument("--doc_file_path", type=str, default='clueweb_data/raw_image_text_new.parquet',
                        help="Path to data file")
    parser.add_argument("--qry_file_path", type=str, default='clueweb_data/anchor.json',
                        help="Path to data file")
    parser.add_argument("--qry_index_save_file", type=str, default='clueweb_data/saved_array_10.npy',
                        help="Path to data file")
    parser.add_argument('--output_path',type=str,default='clueweb_data/clueweb_data.parquet')
    args = parser.parse_args()

    load_index_numpy = np.load(args.qry_index_save_file)
    load_index_data = load_index_numpy.tolist()
    all_data=[]
    doc_data = pd.read_parquet(args.doc_file_path)
    with open(args.qry_file_path, 'r') as json_file:
        qry_data = json.load(json_file)

    first_filter_data = process_data(qry_data)

    second_filter_query_list=[]
    for qry_id in load_index_data:
        query = first_filter_data[qry_id]
        second_filter_query_list.append(query)

    doc_list = []
    for doc_id in range(doc_data.shape[0]):
        doc_example = doc_data.iloc[doc_id]
        doc_example = doc_example.to_dict()
        doc_list.append(doc_example)
    count = 0
    for second_filter_query in tqdm(second_filter_query_list):
        label = second_filter_query['label']
        for one_doc in doc_list:
            if label == one_doc["cw22id"]:
                tmp=one_doc.copy()
                tmp['query']=second_filter_query['query']
                tmp['qid']='q_'+str(count)
                tmp['text_id']='text_'+str(count)
                tmp['img_id']='img_'+str(count)
                all_data.append(tmp)
                count+=1
                break
                print("----------------")
    all_data=pd.DataFrame(all_data)

    print("num after filter:", all_data.shape[0])

    # output_file = f'first_filter_img_text_{args.file_num}.parquet'
    # output_file_path = os.path.join(args.output_root_path, output_file)
    table = pa.Table.from_pandas(all_data)
    pq.write_table(table, args.output_path)



    print("-------------------------------")

        # for item in all_data:
        #     json_file.write(json.dumps(item)+'\n')








    print("------------------------")

if __name__ == '__main__':
    main()


