import os
import argparse
import pprint
import pandas as pd
import torch

from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, EarlyStoppingCallback, Trainer, TrainingArguments
from sklearn.metrics import f1_score

location = '/home/mudang/script/gen_openmp/'
data_location = location + 'data/'

def compute_metrics( pred ):
    labels = pred.label_ids
    preds = pred.prediction.argmax(-1)

    # em = exact match
    em = sum( [ 1 if p == l else 0 for p, l in zip( preds, labels ) ]  ) / len( labels )

    f1 = f1_score( labels, preds, average='marco' )
    
    return {
        'f1': f1,
        'exact_match': em
    }

def run_training( args, model, train_data, eval_data ):
    # do somethings

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,

        load_best_model_at_end=True
    )

    trainer = Trainer( 
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[ EarlyStoppingCallback(early_stopping_patience=args.early_stop) ]
    )

    trainer.train()

    results = trainer.evaluate()

    print( results )

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')

def load_tokenize_data(args, device):
    # Load and tokenize data
    # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
    # datasets = load_dataset("code_x_glue_ct_code_to_text", 'python', split="train")
    df = pd.read_pickle( args.instruct_data_path )
    datasets = Dataset.from_pandas( df )
    ev_df = pd.read_pickle( args.evaluate_data_path )
    evsets = Dataset.from_pandas( ev_df )
    tokenizer = AutoTokenizer.from_pretrained(args.load)

    def preprocess_function(examples):
        # source = [' '.join(ex) for ex in examples["code_tokens"]]
        # target = [' '.join(ex) for ex in examples["docstring_tokens"]]
        source = examples['source']
        target = examples['target']

        model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
        labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"].copy()
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
        ]
        return model_inputs

    train_data = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets.column_names,
        num_proc=4,
        load_from_cache_file=False,
    )

    eval_data = evsets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets.column_names,
        num_proc=4,
        load_from_cache_file=False
    )
    print(f'  ==> Loaded {len(train_data)} samples')
    train_data.save_to_disk(args.cache_data)
    print(f'  ==> Saved to {args.cache_data}')
    eval_data.save_to_disk( args.eval_data )
    return train_data, eval_data

def main( args ):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if( os.path.exists( args.save_dir ) ):
        os.makedirs( args.save_dir )
    if( os.path.exists( args.cache_data ) ):
        os.makedirs( args.cache_data )
    if( os.path.exists( args.eval_data ) ):
        os.makedirs( args.eval_data )

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data, eval_data = load_tokenize_data(args, device)

    # if args.data_num != -1:
    #     train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModel.from_pretrained(args.load, trust_remote_code=True).to(device)
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, train_data, eval_data)

if __name__ == "__main__":
    # df = pd.read_pickle( data_location + '/data202310_stack_v3.pkl' )

    # data = df['for_raw']
    # label = df['omp_raw']

    # ds = Dataset.from_dict( {'data': data, 'label': label}).with_format('torch')

    # dataloader = DataLoader( ds, batch_size=4 )
    # for batch in dataloader:
    #     print( batch )
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--instruct-data-path', default='../data/training_data.pkl', type=str)
    parser.add_argument('--evaluate-data-path', default='../data/eval_data.pkl', type=str)
    parser.add_argument('--cache-data', default='cache_data/train_cache', type=str)
    parser.add_argument('--eval-data', default='cache_data/eval_cache', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m-bimodal', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--early_stop', default=4, type=int)

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/openmp_gen", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    # os.makedirs(args.save_dir, exist_ok=True)

    main(args)
    # training_data = load_tokenize_data( args )

    # print( training_data )