#  MARVEL: Unlocking the Multi-Modal Capability of Dense Retrieval via Visual Module Plugin
Source code for our paper : [MARVEL: Unlocking the Multi-Modal Capability of Dense Retrieval via Visual Module Plugin](https://arxiv.org/abs/2310.14037)

Click the links below to view our papers and checkpoints

<a href='https://arxiv.org/abs/2310.14037'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a><a href='https://huggingface.co/OpenMatch/marvel-pretrain'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrain-blue'><a href='https://huggingface.co/OpenMatch/marvel-dpr-webqa'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-WebQA_DPR-blue'></a><a href='https://huggingface.co/OpenMatch/marvel-dpr-clueweb'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClueWeb22_MM_DPR-blue'></a><a href='https://huggingface.co/OpenMatch/marvel-ance-webqa'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-WebQA_ANCE-blue'></a><a href='https://huggingface.co/OpenMatch/marvel-ance-clueweb'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ClueWeb22_MM_ANCE-blue'></a>

 If you find this work useful, please cite our paper  and give us a shining star 🌟

## Overview

MARVEL unlocks the multi-modal capability of dense retrieval via visual module plugin. It encodes queries and multi-modal documents with a unified encoder model to bridge the modality gap between images and texts, conducts retrieval, modality routing, and result fusion within a unified embedding space.

<p align="center">
  <img align="middle" src="image/marvel.gif" height="350" alt="MARVEL"/>
</p>

## Requirement
**1. Install the following packages using Pip or Conda under this environment**

```
Python==3.7
Pytorch
transformers
clip
faiss-cpu==1.7.0
tqdm
numpy
base64
Install the pytrec_eval from https://github.com/cvangysel/pytrec_eval
```
We provide the version file `requirements.txt` of all our used packages, if you have any problems configuring the environment, please refer to this document.

**2. Prepare the pretrained CLIP and T5-ANCE**

MARVEL is built on [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) and [T5-ANCE](https://huggingface.co/OpenMatch/t5-ance) model.

## Reproduce MARVEL
### Download Code & Dataset
* First, use `git clone` to download this project:
```bash
git clone https://github.com/OpenMatch/MARVEL
cd MARVEL
```
* Download link for our WebQA: [WebQA](https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/data.zip). If you want to use our ClueWeb22-MM and pretrain data, please obtain ClueWeb license first and contact us by email.
* Please make sure that the files under the data folder contain the following before running:
```
data/
├──WebQA/
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── test_qrels.txt
│   ├── all_docs.json
│   ├── all_imgs.json
│   ├── imgs.tsv
│   └── imgs.lineidx.new
├──ClueWeb22-MM/
│   ├── train.parquet
│   ├── dev.parquet
│   ├── test.parquet
│   ├── test_qrels.txt
│   ├── text.parquet
│   └── image.parquet
└──pretrain/
    ├── train.parquet
    └── dev.parquet
```
### Train MARVEL-ANCE
**Using the WebQA dataset as an example, I will show you how to reproduce the results in the MARVEL paper. The same is true for the ClueWeb22-MM dataset.**

* First step: Go to the ``pretrain`` folder and pretrain MARVEL's visual module:
```
cd pretrain
bash train.sh
```
* Second step: Go to the ``DPR`` folder and train MARVEL-DPR using inbatch negatives:
```
cd DPR
bash train_webqa.sh
```
* Third step: Then using MERVEL-DPR to generate hard negatives for training MARVEL-ANCE: 
```
bash get_hn_webqa.sh
```
* Final step: Go to the ``ANCE`` folder and train MARVEL-ANCE using hard negatives: 
```
cd ANCE
bash train_ance_webqa.sh
```

## Evaluate Retrieval Effectiveness
* These experimental results are shown in Table 2 of our paper.
* Go to the ``DPR`` or ``ANCE`` folder and evaluate model performance as follow:
```
bash gen_embeds.sh
bash retrieval.sh
```



## Results
The results are shown as follows.
- WebQA

| Setting             | Model                               | MRR@10 | NDCG@10 | Rec@100 | 
|------------------------------|----------------------------------------------|:------:|:-------:|:-------:|
| Single Modality\\(Text Only) | BM25                                         | 53.75  |  49.60  |  80.69  |
|                              | DPR (Zero-Shot)   | 22.72  |  20.06  |  45.43  |
|                              | CLIP-Text (Zero-Shot) | 18.16  |  16.76  |  39.83  |
|                              | Anchor-DR (Zero-Shot) | 39.96  |  37.09  |  71.32  |
|                              | T5-ANCE (Zero-Shot)   | 41.57  |  37.92  |  69.33  |
|                              | BERT-DPR          | 42.16  |  39.57  |  77.10  |
|                              | NQ-DPR            | 41.88  |  39.65  |  42.44  |
|                              | NQ-ANCE         | 45.54  |  42.05  |  69.31  |
| Divide-Conquer               | VinVL-DPR                                    | 22.11  |  22.92  |  62.82  |
|                              | CLIP-DPR                                     | 37.35  |  37.56  |  85.53  |
|                              | BM25 & CLIP-DPR                             | 42.27  |  41.58  |  87.50  |
| UnivSearch                   | CLIP (Zero-Shot)                             | 10.59  |  8.69   |  20.21  |
|                              | VinVL-DPR                                    | 38.14  |  35.43  |  69.42  |
|                              | CLIP-DPR                                     | 48.83  |  46.32  |  86.43  |
|                              | UniVL-DR                                     | 62.40  |  59.32  |  89.42  |
|                              | MARVEL-DPR                                   | 55.71  |  52.94  |  88.23  |
|                              | MARVEL-ANCE                                  | 65.15  |  62.95  |  92.40  |

- ClueWeb22-MM
  
| Setting             | Model                               | MRR@10 | NDCG@10 | Rec@100 |
|------------------------------|----------------------------------------------|:------:|:-------:|:-------:|
| Single Modality\\(Text Only) | BM25                                         | 40.81  |  46.08  |  78.22  |
|                              | DPR (Zero-Shot)   | 20.59  |  23.24  |  44.93  |
|                              | CLIP-Text (Zero-Shot) | 30.13  |  33.91  |  59.53  |
|                              | Anchor-DR (Zero-Shot) | 42.92  |  48.50  |  76.52  |
|                              | T5-ANCE (Zero-Shot)   | 45.65  |  51.71  |  83.23  |
|                              | BERT-DPR          | 38.56  |  44.41  |  80.38  |
|                              | NQ-DPR            | 42.35  |  61.71  |  83.50  |
|                              | NQ-ANCE         | 45.89  |  51.83  |  81.21  |
| Divide-Conquer               | VinVL-DPR                                    | 29.97  |  36.13  |  74.56  |
|                              | CLIP-DPR                                     | 39.54  |  47.16  |  87.25  |
|                              | BM25 & CLIP-DPR                             | 41.58  |  48.67  |  83.50  |
| UnivSearch                   | CLIP (Zero-Shot)                             | 16.28  |  18.52  |  40.36  |
|                              | VinVL-DPR                                    | 35.09  |  40.36  |  75.06  |
|                              | CLIP-DPR                                     | 42.59  |  49.24  |  87.07  |
|                              | UniVL-DR                                     | 47.99  |  55.41  |  90.46  |
|                              | MARVEL-DPR                                   | 46.93  |  53.76  |  88.74  |
|                              | MARVEL-ANCE                                  | 55.19  |  62.83  |  93.16  |




## Contact
If you have questions, suggestions, and bug reports, please email:
```
zhoutianshuo@stumail.neu.edu.cn     meisen@stumail.neu.edu.cn  
```
