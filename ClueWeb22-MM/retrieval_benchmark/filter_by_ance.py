import argparse
import json

import faiss
import numpy as np
import pandas as pd
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


class DocumentDataset(Dataset):
    def __init__(self, data,tokenizer,args):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.data.iloc[index]
        example = example.to_dict()
        text = example['alt']
        label = example['cw22id']
        item = {}
        item['label'] = label
        item['text'] = text
        return item

    def collect_fn(self, data):

        batch_doc = []

        batch_target = []
        for example in data:
            label = example['label']
            doc_text = example['text']
            batch_target.append(label)
            batch_doc.append(doc_text)

        outputs = self.tokenizer.batch_encode_plus(
            batch_doc,
            max_length=self.args.doc_maxlen,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        return {
                    "doc_ids": input_ids,
                    "doc_masks": attention_mask,
                    "label": batch_target,
                }

class QueryDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        super().__init__()
        self.data = self.process_data(data)
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        example = self.data[index]

        return example
    def process_data(self,data):
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
                all_data.append({'query':query_text,'label':cw22id})
        return all_data

    def collect_fn(self, data):
        batch_qry = []

        batch_target = []
        for example in data:
            label = example['label']
            query_text = example['query']
            batch_target.append(label)
            batch_qry.append(query_text)
        outputs = self.tokenizer.batch_encode_plus(
            batch_qry,
            max_length=self.args.qry_maxlen,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True,
        )
        input_ids = outputs["input_ids"]
        attention_mask = outputs["attention_mask"]
        return {
            "qry_ids": input_ids,
            "qry_masks": attention_mask,
            "label":batch_target,
        }


def retrieve(model, qry_dataloader, doc_dataloader, device, args):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    doc_emb_list = []
    qry_emb_list = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(doc_dataloader)):
            doc_inputs = batch["doc_ids"].to(device)
            doc_masks = batch["doc_masks"].to(device)
            batch_target = batch["label"]

            decoder_input_ids = torch.zeros((doc_inputs.shape[0], 1), dtype=torch.long).to(doc_inputs.device)
            doc_output = model(input_ids=doc_inputs, attention_mask=doc_masks, decoder_input_ids=decoder_input_ids,
                               output_hidden_states=True, return_dict=True)
            doc_emb = doc_output.last_hidden_state[:, 0, :]
            doc_emb = doc_emb.detach().cpu().numpy()


        for i, batch in tqdm(enumerate(qry_dataloader)):
            qry_inputs = batch["qry_ids"].to(device)
            qry_masks = batch["qry_masks"].to(device)
            batch_target = batch["label"]

            decoder_input_ids = torch.zeros((qry_inputs.shape[0], 1), dtype=torch.long).to(qry_inputs.device)
            qry_output = model(input_ids=qry_inputs, attention_mask=qry_masks,decoder_input_ids=decoder_input_ids,
                              output_hidden_states=True, return_dict=True)
            qry_emb = qry_output.last_hidden_state[:, 0, :]
            qry_emb = qry_emb.detach().cpu().numpy()
            for ids in range(len(batch_target)):
                query_dict = {}
                query_dict[batch_target[ids]] = qry_emb[ids,:]
                qry_emb_list.append(query_dict)




            for ids in range(len(batch_target)):
                doc_dict = {}
                doc_dict[batch_target[ids]] = doc_emb[ids,:]
                doc_emb_list.append(doc_dict)




        with open(args.qry_save_file, mode='wb') as f:
            pickle.dump(qry_emb_list, f)

        with open(args.doc_save_file, mode='wb') as f:
            pickle.dump(doc_emb_list, f)


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--doc_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--qry_maxlen', type=int, default=128,
                        help='maxlen')
    parser.add_argument('--model_path', type=str, default='/OpenMatch/t5-ance',
                        help='model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument("--doc_file_path", type=str, default='clueweb_data/raw_image_text_new.parquet',
                        help="Path to data file")

    parser.add_argument("--qry_file_path", type=str, default='clueweb_data/anchor.json',
                        help="Path to data file")

    parser.add_argument("--qry_save_file", type=str, default='clueweb_data/qry_embedding.json',
                        help="Path to data file")
    parser.add_argument("--doc_save_file", type=str, default='clueweb_data/doc_embedding.json',
                        help="Path to data file")

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(device)

    # load document
    doc_data = pd.read_parquet(args.doc_file_path)

    doc_dataset = DocumentDataset(doc_data, tokenizer, args)
    doc_sampler = SequentialSampler(doc_dataset)
    doc_dataloader = DataLoader(
        doc_dataset,
        sampler=doc_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=doc_dataset.collect_fn
    )

    # load query
    with open(args.qry_file_path, 'r') as json_file:
        qry_data = json.load(json_file)
    qry_dataset = QueryDataset(qry_data, tokenizer, args)
    qry_sampler = SequentialSampler(qry_dataset)
    qry_dataloader = DataLoader(
        qry_dataset,
        sampler=qry_sampler,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=qry_dataset.collect_fn
    )

    retrieve(model, qry_dataloader, doc_dataloader, device,args)





if __name__ == '__main__':
    main()