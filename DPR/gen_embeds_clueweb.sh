export CUDA_VISIBLE_DEVICES=3

python gen_embeddings.py --out_path checkpoint_multi_inb_clueweb \
--checkpoint checkpoint_multi_inb_clueweb/model.best.pt \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--cap_path ../data/ClueWeb22-MM/image.parquet \
--query_path ../data/ClueWeb22-MM/test.parquet \
--t5_model_name OpenMatch/t5-ance \
--max_text_len 128 \
--encode_query \
--encode_img \
--encode_txt