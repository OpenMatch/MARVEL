export CUDA_VISIBLE_DEVICES=3

python gen_embeddings.py --out_path checkpoint_multi_inb_webqa \
--checkpoint checkpoint_multi_inb_webqa/model.best.pt \
--img_feat_path ../../data/WebQA/imgs.tsv \
--img_linelist_path ../../data/WebQA/imgs.lineidx.new \
--doc_path ../../data/WebQA/all_docs.json \
--cap_path ../../data/WebQA/all_imgs.json \
--query_path ../../data/WebQA/train.json \
--max_text_len 128 \
--encode_img \
--encode_txt

python gen_embeddings.py --out_path checkpoint_multi_inb_webqa/tmp \
--checkpoint checkpoint_multi_inb_webqa/model.best.pt \
--img_feat_path ../../data/WebQA/imgs.tsv \
--img_linelist_path ../../data/WebQA/imgs.lineidx.new \
--doc_path ../../data/WebQA/all_docs.json \
--cap_path ../../data/WebQA/all_imgs.json \
--query_path ../../data/WebQA/train.json \
--max_text_len 128 \
--encode_query \


mv ./checkpoint_multi_inb_webqa/tmp/query_embedding.pkl ./checkpoint_multi_inb_webqa/train_query_embedding.pkl

python gen_embeddings.py --out_path checkpoint_multi_inb_webqa/tmp \
--checkpoint checkpoint_multi_inb_webqa/model.best.pt \
--img_feat_path ../../data/WebQA/imgs.tsv \
--img_linelist_path ../../data/WebQA/imgs.lineidx.new \
--doc_path ../../data/WebQA/all_docs.json \
--cap_path ../../data/WebQA/all_imgs.json \
--query_path ../../data/WebQA/dev.json \
--max_text_len 128 \
--encode_query

mv ./checkpoint_multi_inb_webqa/tmp/query_embedding.pkl ./checkpoint_multi_inb_webqa/dev_query_embedding.pkl

python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb_webqa/train_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_webqa/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb_webqa/txt_embedding.pkl \
--data_path ../../data/WebQA/train.json \
--out_path ./checkpoint_multi_inb_webqa/train_all.json

python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb_webqa/dev_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_webqa/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb_webqa/txt_embedding.pkl \
--data_path ../../data/WebQA/dev.json \
--out_path ./checkpoint_multi_inb_webqa/dev_all.json



