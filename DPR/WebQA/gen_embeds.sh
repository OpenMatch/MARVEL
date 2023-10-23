export CUDA_VISIBLE_DEVICES=3

python gen_embeddings.py --out_path checkpoint_multi_inb_webqa \
--checkpoint checkpoint_multi_inb_webqa/model.best.pt \
--img_feat_path ../../data/WebQA/imgs.tsv \
--img_linelist_path ../../data/WebQA/imgs.lineidx.new \
--doc_path ../../data/WebQA/all_docs.json \
--cap_path ../../data/WebQA/all_imgs.json \
--query_path ../../data/WebQA/test.json \
--max_text_len 128 \
--encode_query \
--encode_img \
--encode_txt