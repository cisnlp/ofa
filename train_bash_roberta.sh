WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=6,7 torchrun --rdzv_endpoint=0.0.0.0:29400 --nproc_per_node=2 ../run_extra.py \
  --model_name_or_path roberta-base \
  --train_file /mounts/data/proj/ayyoobbig/1000LM/data/1000LM.txt \
  --tokenizer_name /mounts/data/proj/ayyoobbig/1000LM/tokenizer/1000LM_extended_spm \
  --output_dir /mounts/data/proj/ayyoobbig/ofa/trained_models/updated/LM_ofa \
  --cache_dir /mounts/data/proj/ayyoobbig/ofa/cache \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 16 \
  --fp16 True \
  --do_train \
  --num_train_epochs 100 \
  --save_steps 10000 \
  --ddp_timeout 259200 \
  --use_initialization True \
  --random_initialization False \
  --num_primitive 200 \
  --embedding_path /mounts/data/proj/yihong/newhome/OFA/stored_factorization/updated \
  --only_eng_vocab False \
  --preprocessing_num_workers 8 \
  --ignore_data_skip True