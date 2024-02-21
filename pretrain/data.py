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
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_IMAGE_TOKEN = "<image>"
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
        self.cap_len=args.cap_len
        self.only_image_caption_contrastive_loss=args.only_image_caption_contrastive_loss

        self.data = data


    def __len__(self):
        return len(self.data)

    def encode_img(self, caption,img):
        img = self.preprocess_fn(images=Image.open(io.BytesIO(img)), return_tensors="pt")["pixel_values"][0]
        if caption != None:
            if self.only_image_caption_contrastive_loss:
                return {'img': img, 'cap':caption}
            pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap=pre_token+" "+"caption: "+ caption
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        processed_batch = {}
        for index, example in enumerate(batch):
            img_inputs.append(example['image_doc']['img'])
            cap_inputs.append(example['image_doc']['cap'])
            txt_inputs.append(example['pos_txt'])


        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        if len(cap_inputs) != 0:
            assert len(cap_inputs) == len(img_inputs)
            processed_batch['cap_inputs'] = self.tokenizer(cap_inputs, return_tensors='pt',max_length=self.cap_len,padding='max_length',truncation=True)

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = self.tokenizer(txt_inputs, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        instance = {}
        caption=example['alt']
        img=example['BUFFER']
        instance["image_doc"] = self.encode_img(caption,img)
        instance["pos_txt"] = example['surround']
        return instance

def load_file(path):
    data = []
    input_data = pd.read_parquet(path)
    for index in range(len(input_data)):
        example = input_data.iloc[index]
        example = example.to_dict()
        image_caption=example['alt']
        caption_list=image_caption.split()
        if len(caption_list)<2:
            continue
        data.append(example)
    return data





