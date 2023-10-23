export CUDA_VISIBLE_DEVICES=0

python train.py  --out_path ./checkpoint_multi_inb_clueweb/ \
--train_path ../../data/ClueWeb22-MM/train.parquet \
--valid_path ../../data/ClueWeb22-MM/dev.parquet \
--t5_model_name OpenMatch/t5-ance \
--text_len 128 \
--freeze_vision_model


