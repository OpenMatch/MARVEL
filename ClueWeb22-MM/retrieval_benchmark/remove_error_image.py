"""
Use clip model to remove image data that can't be processed properly
"""

import argparse
import io
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import clip
import torch

def get_one_doc_example(args,doc_path):

    model, preprocess = clip.load(args.clip_model, device=args.device)
    input_data = pd.read_parquet(doc_path)
    count=0
    total = len(input_data)
    a=[]
    for index in tqdm(range(total)):
        example = input_data.iloc[index]
        example = example.to_dict()
        image_buffer = io.BytesIO(example['img'])
        img = Image.open(image_buffer)
        if preprocess is not None:
            try:
                img = preprocess(img)
            except Exception as e:
                print("Error: the picture is error", e)
                input_data.drop(index,inplace=True)
                count+=1
                print('---------index--------')
                print(index)
                print(example['img_id'])
                a.append(index)
    print(count)
    print(a)
    return input_data




def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--doc_file_path", type=str, default='clueweb-imgtext/thirdfilter/combined_data.parquet', help="Path to input data file")
    parser.add_argument("--output_file_path", type=str, default='clueweb_data/raw_image_text_new.parquet', help="Path to output data file")
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        args.device = torch.device(args.device)

    doc_data = get_one_doc_example(args,args.doc_file_path)
    doc_data.to_parquet(args.output_file_path)



if __name__ == '__main__':
    main()