python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_webqa/query_embedding.pkl \
--img_embed_path ./checkpoint_multi_hn_webqa/img_embedding.pkl \
--qrel_path ../../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_hn_webqa

python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_webqa/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_hn_webqa/txt_embedding.pkl \
--img_embed_path ./checkpoint_multi_hn_webqa/img_embedding.pkl \
--qrel_path ../../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_hn_webqa


python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_webqa/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_hn_webqa/txt_embedding.pkl \
--qrel_path ../../data/WebQA/test_qrels.txt \
--out_path ./checkpoint_multi_hn_webqa
