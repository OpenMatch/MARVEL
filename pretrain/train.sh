export CUDA_VISIBLE_DEVICES=0

python train.py  --out_path ./checkpoint_pretrain/ \
--train_path ../data/pretrain/train.parquet \
--valid_path ../data/pretrain/dev.parquet \
--cap_len 64 \
--freeze_language_model \
--only_image_caption_contrastive_loss