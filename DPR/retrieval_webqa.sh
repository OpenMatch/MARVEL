python retrieval.py   --query_embed_path  ./checkpoint_multi_inb_webqa/query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_webqa/img_embedding.pkl \
--data_path ../data/WebQA/test.json \
--qrel_path ../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_inb_webqa

python retrieval.py   --query_embed_path  ./checkpoint_multi_inb_webqa/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_inb_webqa/txt_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb_webqa/img_embedding.pkl \
--data_path ../data/WebQA/test.json \
--qrel_path ../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_inb_webqa

python retrieval.py   --query_embed_path  ./checkpoint_multi_inb_webqa/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_inb_webqa/txt_embedding.pkl \
--data_path ../data/WebQA/test.json \
--qrel_path ../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_inb_webqa