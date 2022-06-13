#!/bin/sh
export WANDB_PROJECT=instruct_dial-t0
export WANDB_DISABLED=true
cd scripts
module load gcc-7.4
#rm -r /home/prakharg/.cache/huggingface/modules/datasets_modules/
deepspeed ./run_traint0.py \
    --model_name_or_path bigscience/T0_3B \
    --do_train \
    --do_eval \
    --train_file text2textfiles/traindata_eval_train3_t5_d3.json \
    --validation_file text2textfiles/valid_data.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./tmp/topp-gs72lin_eval_train3_t5_d3.json \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps 36\
    --gradient_checkpointing \
    --learning_rate 5e-05 \
    --deepspeed t0ds-config.json \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 10\
    --bf16\
    --evaluation_strategy steps\
    --num_train_epochs 2\
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 200\
    --eval_steps 200\
    --logging_steps 25\ 
