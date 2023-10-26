import json
from visual import TSVFile
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import base64
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import clip
from data import load_caps
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

logger = logging.getLogger()
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import load_model,get_img_patch_token_size

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class WebQAImgDataset(Dataset):
    def __init__(self, args, preprocess, tokenizer, max_text_len, captions=None):

        self.img_map = {}
        self.img_ids = []
        self.captions = captions
        self.preprocess_fn = preprocess
        self.tokenizer = tokenizer
        self.text_len = max_text_len
        self.img_special_len = args.img_patch_token_size
        self.cap_len = self.text_len+self.img_special_len+2

        all_img_num = 0
        with open(args.img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
                self.img_ids.append(tokens[0])
        self.img_tsv = TSVFile(args.img_feat_path, all_img_num)

    def __len__(self):
        return len(self.img_ids)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(images=Image.open(io.BytesIO(base64.b64decode(img))), return_tensors="pt")["pixel_values"][0]
        if self.captions != None:
            cap = self.captions[idx]
            pre_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap = pre_token + " " + "caption: " + cap
            return {'img': img, 'cap': cap}
        return {'img': img}

    def Collector(self, batch):
        img_inputs = []
        img_caps = []
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            if 'img_caps' in example:
                img_caps.append(example['img_caps'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
        if len(img_caps) != 0:
            processed_batch['img_caps'] = self.tokenizer(img_caps, return_tensors='pt', max_length=self.cap_len,
                                                         padding='max_length',
                                                         truncation=True)
        return processed_batch

    def __getitem__(self, index):
        img_idx = self.img_ids[index]
        img_inputs = self.encode_img(img_idx)
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if 'cap' in img_inputs:
            instance['img_caps'] = img_inputs['cap']

        return instance


class ClueWebImgDataset(Dataset):
    def __init__(self, args, preprocess, tokenizer, max_text_len, img_doc=None):

        self.img_doc = img_doc
        self.preprocess_fn = preprocess
        self.tokenizer = tokenizer
        self.text_len = max_text_len
        self.img_special_len = args.img_patch_token_size
        self.cap_len = self.text_len+self.img_special_len+2


    def __len__(self):
        return len(self.img_doc)

    def encode_img(self, caption,img):
        img = self.preprocess_fn(images=Image.open(io.BytesIO(img)), return_tensors="pt")["pixel_values"][0]
        if caption!= None:
            pre_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap = pre_token + " " + "caption: " + caption
            return {'img': img, 'cap': cap}
        return {'img': img}

    def Collector(self, batch):
        img_inputs = []
        img_caps = []
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            if 'img_caps' in example:
                img_caps.append(example['img_caps'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
        if len(img_caps) != 0:
            processed_batch['img_caps'] = self.tokenizer(img_caps, return_tensors='pt', max_length=self.cap_len,
                                                         padding='max_length',
                                                         truncation=True)
        return processed_batch

    def __getitem__(self, index):
        example = self.img_doc.iloc[index]
        example = example.to_dict()
        img_idx = example['img_id']
        caption = example['caption']
        img = example['img']
        img_inputs = self.encode_img(caption,img)
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if 'cap' in img_inputs:
            instance['img_caps'] = img_inputs['cap']

        return instance


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_text_len):
        self.data = data
        self.text_len = max_text_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def Collector(self, batch):
        txt_inputs = []
        idx_list = []

        for qid, example in enumerate(batch):
            txt_inputs.append(example['txt_inputs'])
            idx_list.append(example['idx'])
        processed_batch = {
            'txt_inputs': self.tokenizer(txt_inputs, return_tensors='pt', max_length=self.text_len,
                                         padding='max_length', truncation=True),
            'idx_list': idx_list
        }
        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        txt_inputs = example[1]

        return {
            'idx': example[0],
            'txt_inputs': txt_inputs
        }


def gen_embeddings(model, valid_reader, outpath):
    model.eval()
    all_embeddings = []
    all_index = []
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            idx_list = batch['idx_list']
            if 'img_inputs' in batch:
                if 'img_inputs' in batch and 'img_caps' in batch:
                    embeddings = model(batch['img_inputs'].cuda(), batch['img_caps'], device)
                elif 'img_inputs' in batch and 'img_caps' not in batch:
                    embeddings = model(batch['img_inputs'].cuda(), None, device)
            else:
                embeddings = model(None, batch["txt_inputs"], device)
            embeddings = F.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu()
            assert len(embeddings) == len(idx_list)
            all_index.extend(idx_list)
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    with open(outpath, 'wb') as fout:
        pickle.dump((all_index, all_embeddings), fout)

def load_img_docs(path):
    input_data = pd.read_parquet(path)
    return input_data


def load_docs(path):
    data = []
    if "clueweb" in path.lower():
        input_data = pd.read_parquet(path)
        for index in range(len(input_data)):
            example = input_data.iloc[index]
            example = example.to_dict()
            did = example['text_id']
            data.append([did, example['surround']])
    elif "webqa" in path.lower():
        with open(path) as fin:
            for line in fin:
                example = json.loads(line.strip())
                did = str(example['snippet_id'])
                data.append([did, example['fact']])
    else:
        raise ("The path of data is error!")
    return data


def load_queries(path):
    data = []
    if "clueweb" in path.lower():
        input_data = pd.read_parquet(path)
        for index in range(len(input_data)):
            example = input_data.iloc[index]
            qid = str(example['qid'])
            data.append([qid, example['query']])
    elif "webqa" in path.lower():
        with open(path) as fin:
            for line in fin:
                example = json.loads(line.strip())
                qid = str(example['qid'])
                data.append([qid, example['Q']])
    else:
        raise ("The path of data is error!")
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--t5_model_name",type=str,default='OpenMatch/t5-ance')
    parser.add_argument("--clip_model_name",type=str,default='openai/clip-vit-base-patch32')
    parser.add_argument('--select_layer',type=int,default='-1')

    parser.add_argument("--out_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--img_feat_path", type=str)
    parser.add_argument("--img_linelist_path", type=str)

    parser.add_argument("--query_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--cap_path", type=str)
    parser.add_argument("--img_doc_path", type=str)

    parser.add_argument('--encode_txt', action='store_true', default=False)
    parser.add_argument('--encode_img', action='store_true', default=False)
    parser.add_argument('--encode_query', action='store_true', default=False)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    t5_tokenizer, model, image_processor = load_model(args,device)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.cuda()

    args.img_patch_token_size = get_img_patch_token_size(args.clip_model_name)

    docs = load_docs(args.doc_path)
    if args.encode_query:
        queries = load_queries(args.query_path)
        query_data = TextDataset(queries, t5_tokenizer, args.max_text_len)
        query_sampler = SequentialSampler(query_data)
        query_reader = DataLoader(dataset=query_data, sampler=query_sampler, num_workers=args.num_workers,
                                  batch_size=args.batch_size, collate_fn=query_data.Collector)

        output = os.path.join(args.out_path, 'query_embedding.pkl')
        gen_embeddings(model, query_reader, output)

    if args.encode_img:
        if "clueweb" in args.cap_path.lower():
            img_doc=None
            if args.img_doc_path:
                img_doc = load_img_docs(args.img_doc_path)
            img_data = ClueWebImgDataset(args, image_processor, t5_tokenizer, args.max_text_len,
                                  img_doc=img_doc)
        elif "webqa" in args.cap_path.lower():
            captions = None
            if args.cap_path:
                captions = load_caps(args.cap_path)
            img_data = WebQAImgDataset(args, image_processor, t5_tokenizer, args.max_text_len,
                                  captions=captions)
        else:
            raise ("The path of data is error!")
        sampler = SequentialSampler(img_data)
        img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=img_data.Collector)
        output = os.path.join(args.out_path, 'img_embedding.pkl')
        gen_embeddings(model, img_reader, output)

    if args.encode_txt:
        docs = load_docs(args.doc_path)
        txt_data = TextDataset(docs, t5_tokenizer, args.max_text_len)
        txt_sampler = SequentialSampler(txt_data)
        txt_reader = DataLoader(dataset=txt_data, sampler=txt_sampler, num_workers=args.num_workers,
                                batch_size=args.batch_size, collate_fn=txt_data.Collector)

        output = os.path.join(args.out_path, 'txt_embedding.pkl')
        gen_embeddings(model, txt_reader, output)
