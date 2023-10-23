import json
import os
# from visual import TSVFile
import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
from PIL import ImageFile
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"



class ClueWebDataset(Dataset):
    def __init__(self, args, preprocess_fn, tokenizer, data, shuffle,img_special_len=49):
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn
        self.tokenizer=tokenizer

        self.img_map = {}
        self.img_tsv = []
        self.img_special_len=img_special_len

        self.text_len = args.text_len
        self.cap_len = img_special_len+2+self.text_len

        self.data = data


    def __len__(self):
        return len(self.data)


    def encode_img(self, caption,img):

        img = self.preprocess_fn(images=Image.open(io.BytesIO(img)), return_tensors="pt")["pixel_values"][0]
        if caption != None:
            pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap=pre_token+" "+"caption: "+ caption
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        txt_labels = []
        img_labels = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            if 'pos_img' in example:
                img_inputs.append(example['pos_img']['img'])
                if 'cap' in example['pos_img']:
                    cap_inputs.append(example['pos_img']['cap'])
                img_labels.append(qid)
            if 'pos_txt' in example:
                txt_inputs.append(example['pos_txt'])
                txt_labels.append(qid)
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    img_inputs.append(instance['img'])
                    if 'cap' in instance:
                        cap_inputs.append(instance['cap'])
                    img_labels.append(-1)
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    txt_inputs.append(instance)
                    txt_labels.append(-1)

        processed_batch['queries'] = self.tokenizer(queries, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
            processed_batch['img_labels'] = img_labels
            if len(cap_inputs) != 0:
                assert len(cap_inputs) == len(img_inputs)
                processed_batch['img_caps'] = self.tokenizer(cap_inputs, return_tensors='pt',max_length=self.cap_len,padding='max_length',truncation=True)

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = self.tokenizer(txt_inputs, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
            processed_batch['txt_labels'] = txt_labels

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['query']
        instance = {'query': query}

        if example['img_id'] != None:
            caption=example['alt']
            img=example['BUFFER']
            instance["pos_img"] = self.encode_img(caption,img)
        elif example['text_id'] != None:
            instance["pos_txt"] = example['surround']
        else:
            raise ('No positive instance!')
        return instance




def load_file(path, txt=True, img=True):
    data = []
    assert (txt or img)
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()

        if txt and example['text_id'] != None:
            data.append(example)
        if img and example['img_id'] != None:
            data.append(example)
    return data


def load_docs(path):
    data = {}
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()
        did = str(example['text_id'])
        data[did] = example['surround']
    return data

def load_caps(path):
    data = {}
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()
        imgid = example['img_id']
        data[imgid] = example['caption']
    return data

