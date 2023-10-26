export CUDA_VISIBLE_DEVICES=0

python gen_embeddings.py --out_path checkpoint_multi_inb_clueweb \
--checkpoint checkpoint_multi_inb_clueweb/model.best.pt \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--query_path ../data/ClueWeb22-MM/train.parquet \
--t5_model_name OpenMatch/t5-ance \
--max_text_len 128 \
--encode_img \
--encode_txt

python gen_embeddings.py --out_path checkpoint_multi_inb_clueweb/tmp \
--checkpoint checkpoint_multi_inb_clueweb/model.best.pt \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--query_path ../data/ClueWeb22-MM/train.parquet \
--t5_model_name OpenMatch/t5-ance \
--max_text_len 128 \
--encode_query \

mv ./checkpoint_multi_inb_clueweb/tmp/query_embedding.pkl ./checkpoint_multi_inb_clueweb/train_query_embedding.pkl

python gen_embeddings.py --out_path checkpoint_multi_inb_clueweb/tmp \
--checkpoint checkpoint_multi_inb_clueweb/model.best.pt \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--query_path ../data/ClueWeb22-MM/dev.parquet \
--t5_model_name OpenMatch/t5-ance \
--max_text_len 128 \
--encode_query \

mv ./checkpoint_multi_inb_clueweb/tmp/query_embedding.pkl ./checkpoint_multi_inb_clueweb/dev_query_embedding.pkl


python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb_clueweb/train_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_clueweb/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb_clueweb/txt_embedding.pkl \
--data_path ../data/ClueWeb22-MM/train.parquet \
--out_path ./checkpoint_multi_inb_clueweb/train_all.parquet

python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb_clueweb/dev_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_clueweb/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb_clueweb/txt_embedding.pkl \
--data_path ../data/ClueWeb22-MM/dev.parquet \
--out_path ./checkpoint_multi_inb_clueweb/dev_all.parquet

