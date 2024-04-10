# ClueWeb22 MM 

## ClueWeb22

ClueWeb22 MM is built on ClueWeb22. ClueWeb22 is the newest in the Lemur Project's ClueWeb line of datasets that support research on information retrieval, natural language processing and related human language technologies. 

The ClueWeb22 datasets are distributed by Carnegie Mellon University for research purposes only. A dataset may be obtained by signing a data license agreement with Carnegie Mellon University. For details on how to get it, please click the following link:

```bash
https://www.lemurproject.org/clueweb22/obtain.php
```


## Quick Start

If you have obtained the copyright license of the ClueWeb22 dataset, you can build ClueWeb22-MM through the following script.

#### Retrieval Benchmark

|Step|Script|Introduction|
|--|:-----|:-----|
|0|gen_image_text_pair.py|Save the url, caption, and surrouding text of the image from the original ClueWeb22 web page|
|1|first_filter_image_text_pair.py|Processing of image_url side: only keep jpg/png/jpeg, and exclude urls containing tlogo, button, icon, plugin, widget|
|2|second_filter.py|Deduplication based on image URL|
|3|third_filter_image_text_pair.py|Merge parquet files|
|4|pre_data.py|Save the url, caption, and surrouding text of the image from the original ClueWeb22 web page|
|5|gen_anchor.py|Generate anchor text as query for each document|
|6|filter_by_ance.py|Use T5-ANCE to encode the query and caption, and filter out the data containing filter_out_keyword_set in the query|
|7|get_trec.py|Keep the retrieved top files|
|8|construct_new_clueweb_data.py|Based on the retrieved top file and filter_out_keyword_set, filter out the data sets for training and testing|
|9|sample_one_label.py|Ensure that each query corresponds to a label, and the text img mode of the label is close to 1:1|
|10|remove_datasets_duplicate.py|Unify the IDs in the data set to prevent the phenomenon of having the same content but different IDs|
|11|split_newdata.py|Partition the data set|
|12|get_qrel.py|Generate qrel file|
|13|get_collection_no_duplicate.py|Remove the data on the training and test set to generate a corpus; and ensure that the data with consistent content and ID are also the same|
|14|new_image.py|The data with index 141666 was removed separately and cannot be opened normally during subsequent data processing.|

#### Pretrain Dataset

|Step|Script|Introduction|
|--|:-----|:-----|
|0|clip_filter.py|Remove the data corresponding to the image that cannot be opened normally, and use VIT/L to calculate the similarity score between image captions|
|1|filter_similarity.py|According to the similarity score, remove data <0.3|
|2|merge_split_datasets.py|Merge and divide data sets|

#### Reproduce

We provide a comparison file containing query, Clueweb ID and image Node ID for easy reproduction. [Link](https://drive.google.com/file/d/1-YpszaRx_sEBLITh_BY_R42SmkGrcvXx/view?usp=sharing)

At the same time, please feel free to contact us if you have any questions about the dataset.
```
zhoutianshuo@stumail.neu.edu.cn     meisen@stumail.neu.edu.cn  
```
