export CUDA_VISIBLE_DEVICES=0

python gen_embeddings.py --out_path checkpoint_multi_hn_clueweb \
--checkpoint ./checkpoint_multi_hn_clueweb/model.best.pt \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--cap_path ../data/ClueWeb22-MM/image.parquet \
--query_path ../data/ClueWeb22-MM/test.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--max_text_len 128 \
--encode_query \
--encode_img \
--encode_txt