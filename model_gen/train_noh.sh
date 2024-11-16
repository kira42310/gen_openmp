deepspeed --num_gpus=1 trainCodeT5p_gen_v1_0_noeval.py \
    --fp16 \
    --instruct-data-path ../data/training_data_filter_token3_ddup.pkl \
    --evaluate-data-path ../data/eval_data_filter_token3_ddup.pkl \
    --deepspeed deepspeed_config.json \
    --epochs $1 \
    --batch-size-per-replica $2 \
    --grad-acc-steps $3 \
    --save_limit $4 \
    --save-dir saved_models/gen_filter_token3_ddup_test
