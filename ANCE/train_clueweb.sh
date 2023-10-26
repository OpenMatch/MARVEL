export CUDA_VISIBLE_DEVICES=0

python train.py  --out_path ./checkpoint_multi_hn_clueweb/ \
--train_path ../DPR/checkpoint_multi_inb_clueweb/train_all.parquet \
--valid_path ../DPR/checkpoint_multi_inb_clueweb/dev_all.parquet \
--doc_path ../data/ClueWeb22-MM/text.parquet \
--cap_path ../data/ClueWeb22-MM/image.parquet \
--img_doc_path ../data/ClueWeb22-MM/image.parquet \
--t5_model_name OpenMatch/t5-ance \
--train_batch_size 64 \
--valid_batch_size 64 \
--text_len 128 \
--pretrained_model_path ../DPR/checkpoint_multi_inb_clueweb/model.best.pt \
--gradient_accumulation_steps 1 \
--img_neg_num 1 \
--txt_neg_num 1 \
--freeze_vision_model