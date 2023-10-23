export CUDA_VISIBLE_DEVICES=3

python train.py  --out_path ./checkpoint_multi_inb_webqa/ \
--train_path ../../data/WebQA/train.json \
--valid_path ../../data/WebQA/dev.json \
--doc_path ../../data/WebQA/all_docs.json \
--cap_path ../../data/WebQA/all_imgs.json \
--img_feat_path ../../data/WebQA/imgs.tsv \
--img_linelist_path ../../data/WebQA/imgs.lineidx.new \
--text_len 128 \
--freeze_vision_model