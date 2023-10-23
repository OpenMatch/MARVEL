import json
import os

import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class ClueWebDataset(Dataset):
    def __init__(self, args, preprocess_fn, tokenizer, data, docs, img_doc,captions, shuffle,img_special_len=49):
        self.neg_num = args.neg_num
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        if not self.shuffle and args.neg_num == 0:
            self.neg_num = self.img_neg_num + self.txt_neg_num
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.data=data
        self.docs = docs
        self.img_doc=img_doc
        self.captions = captions
        self.img_special_len = img_special_len
        self.text_len = args.text_len
        self.cap_len = self.img_special_len+2+self.text_len




    def __len__(self):
        return len(self.data)

    def encode_img(self, idx):
        example = self.img_doc[idx]
        img = example['img']
        caption = example['caption']
        img = self.preprocess_fn(images=Image.open(io.BytesIO(img)), return_tensors="pt")["pixel_values"][0]
        if caption != None:
            pre_token = DEFAULT_IM_START_TOKEN + " " + DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap = pre_token + " " + "caption: " + caption
            return {'img': img, 'cap': cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_inputs = []
        img_dict = {}

        txt_inputs = []
        txt_dict = {}
        cap_inputs = []

        pos_idx = []
        neg_idx = []

        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            if 'pos_img' in example:
                idx = example['pos_img']['idx']
                if idx not in img_dict:
                    img_dict[idx] = len(img_inputs)
                    img_inputs.append(example['pos_img']['img'])
                    if 'cap' in example['pos_img']:
                        cap_inputs.append(example['pos_img']['cap'])
            if 'pos_txt' in example:
                idx = example['pos_txt']['idx']
                if idx not in txt_dict:
                    txt_dict[idx] = len(txt_inputs)
                    txt_inputs.append(example['pos_txt']['txt'])
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    idx = instance['idx']
                    if idx not in img_dict:
                        img_dict[idx] = len(img_inputs)
                        img_inputs.append(instance['img'])
                        if 'cap' in instance:
                            cap_inputs.append(instance['cap'])
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    idx = instance['idx']
                    if idx not in txt_dict:
                        txt_dict[idx] = len(txt_inputs)
                        txt_inputs.append(instance['txt'])

        for qid, example in enumerate(batch):
            if 'pos_img' in example:
                idx = example['pos_img']['idx']
                pos_idx.append(img_dict[idx])
            if 'pos_txt' in example:
                idx = example['pos_txt']['idx']
                pos_idx.append(txt_dict[idx] + len(img_inputs))

            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    idx = instance['idx']
                    neg_idx.append(img_dict[idx])
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    idx = instance['idx']
                    neg_idx.append(txt_dict[idx] + len(img_inputs))

        processed_batch['queries'] = self.tokenizer(queries, return_tensors='pt', max_length=self.text_len,
                                                    padding='max_length', truncation=True)
        processed_batch['pos_idx'] = pos_idx
        processed_batch['neg_idx'] = neg_idx

        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        if len(cap_inputs) != 0:
            assert len(cap_inputs) == len(img_inputs)
            processed_batch['img_caps'] = self.tokenizer(cap_inputs, return_tensors='pt', max_length=self.cap_len,
                                                             padding='max_length', truncation=True)

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = self.tokenizer(txt_inputs, return_tensors='pt', max_length=self.text_len,
                                                           padding='max_length', truncation=True)

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['query']
        instance = {'query': query}

        if example['img_id'] != None:

            idx=example['img_id']
            img = self.encode_img(idx)
            img['idx'] = idx
            instance["pos_img"] = img
        elif example['text_id'] != None:
            idx = example['text_id']
            instance["pos_txt"] = {'idx': idx, 'txt': self.docs[idx]}
        else:
            raise ('No positive instance!')
        if self.neg_num > 0:
            neg_imgs = []
            neg_txts = []
            if 'all_negFacts' in example:
                neg_idx = example['all_negFacts']
            else:
                neg_idx = example['img_negFacts'] + example['txt_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_idx)
            neg_idx = neg_idx[:self.neg_num]

            for idx in neg_idx:
                if idx in self.captions:
                    img = self.encode_img(idx)
                    img['idx'] = idx
                    neg_imgs.append(img)
                if idx in self.docs:
                    neg_txts.append({'idx': idx, 'txt': self.docs[idx]})
            if len(neg_imgs) > 0:
                instance["neg_imgs"] = neg_imgs
            if len(neg_txts) > 0:
                instance["neg_txts"] = neg_txts
        else:
            img_neg_num = self.img_neg_num
            txt_neg_num = self.txt_neg_num
            if len(example['img_negFacts']) < self.img_neg_num:
                img_neg_num = len(example['img_negFacts'])
                txt_neg_num = self.img_neg_num + self.txt_neg_num - img_neg_num
            elif len(example['txt_negFacts']) < self.txt_neg_num:
                txt_neg_num = len(example['txt_negFacts'])
                img_neg_num = self.img_neg_num + self.txt_neg_num - txt_neg_num

            if img_neg_num > 0:
                neg_imgs = []
                neg_img_idx = example['img_negFacts']
                if self.shuffle:
                    np.random.shuffle(neg_img_idx)
                neg_img_idx = neg_img_idx[:img_neg_num]
                for idx in neg_img_idx:
                    img = self.encode_img(idx)
                    img['idx'] = idx
                    neg_imgs.append(img)
                instance["neg_imgs"] = neg_imgs

            if txt_neg_num > 0:
                neg_txts = []
                neg_txt_idx = example['txt_negFacts']
                if self.shuffle:
                    np.random.shuffle(neg_txt_idx)
                neg_txt_idx = neg_txt_idx[:txt_neg_num]
                for idx in neg_txt_idx:
                    neg_txts.append({'idx': idx, 'txt': self.docs[idx]})
                instance["neg_txts"] = neg_txts

        return instance


def load_file(path, txt=True, img=True):
    data = []
    assert (txt or img)
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()

        txt_negFacts = example['txt_negFacts']
        np.random.shuffle(txt_negFacts)
        example['txt_negFacts'] = txt_negFacts

        img_negFacts = example['img_negFacts']
        np.random.shuffle(img_negFacts)
        example['img_negFacts'] = img_negFacts

        if 'all_negFacts' in example:
            all_negFacts = example['all_negFacts']
            np.random.shuffle(all_negFacts)
            example['all_negFacts'] = all_negFacts

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

def load_img_doc(path):
    data={}
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()
        imgid = example['img_id']
        data[imgid] = {'caption':example['caption'],'img':example['img']}
    return data