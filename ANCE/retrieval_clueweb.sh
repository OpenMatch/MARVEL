
python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_clueweb/query_embedding.pkl \
--img_embed_path ./checkpoint_multi_hn_clueweb/img_embedding.pkl \
--data_path ../data/ClueWeb22-MM/test.parquet \
--qrel_path ../data/ClueWeb22-MM/test_qrels.txt \
--out_path ./checkpoint_multi_hn_clueweb

python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_clueweb/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_hn_clueweb/txt_embedding.pkl \
--img_embed_path ./checkpoint_multi_hn_clueweb/img_embedding.pkl \
--data_path ../data/ClueWeb22-MM/test.parquet \
--qrel_path ../data/ClueWeb22-MM/test_qrels.txt \
--out_path ./checkpoint_multi_hn_clueweb

python retrieval.py   --query_embed_path  ./checkpoint_multi_hn_clueweb/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_hn_clueweb/txt_embedding.pkl \
--data_path ../data/ClueWeb22-MM/test.parquet \
--qrel_path ../data/ClueWeb22-MM/test_qrels.txt \
--out_path ./checkpoint_multi_hn_clueweb