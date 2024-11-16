deepspeed --num_gpus=1 trainCodeT5p_gen_v1_0.py \
    --fp16 \
    --instruct-data-path ../data/training_data_filter_token3_ddup.pkl \
    --evaluate-data-path ../data/eval_data_filter_token3_ddup.pkl \
    --epochs $1 \
    --batch-size-per-replica $2 \
    --grad-acc-steps $3 \
    --early_stop 4 \
    --save-dir saved_models/test_data_ddup
