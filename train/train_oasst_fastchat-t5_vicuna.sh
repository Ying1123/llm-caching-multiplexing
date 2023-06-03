python run_glue.py \
  --model_name_or_path bert-base-cased \
  --train_file ../data/oasst_a100_fastchat-t5_vicuna_clean_train.json \
  --validation_file ../data/oasst_a100_fastchat-t5_vicuna_clean_valid.json \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --fp16 True \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --evaluation_strategy "steps" \
  --eval_steps 10 \
  --output_dir models/choice_model_oasst_fastchat-t5_vicuna \
  --overwrite_output_dir

  #--test_file ../data/oasst_a100_fastchat-t5_vicuna_clean_test.json \
