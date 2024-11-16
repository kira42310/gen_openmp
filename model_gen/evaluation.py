from transformers import AutoModel, AutoTokenizer
from datasets import Dataset, load_from_disk, load_dataset
from torch.utils.data import DataLoader
from sctokenizer import CppTokenizer
#from evaluate import load
from evaluate import combine
import pandas as pd
import torch

tk_checkpoint = "Salesforce/codet5p-220m-bimodal"
# checkpoint = "./test/final_checkpoint"
# checkpoint = "./saved_models/gen_filter_token3/"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

raw_path = '../data/eval_data_filter_token3_ddup.pkl'
cache_path = './cache_data/eval_cache/'

# location = "./saved_models/gen_filter_token_noe/"
# list_ckp = [ 'checkpoint-2214', 'checkpoint-4428', 'checkpoint-6642', 'checkpoint-8856', 'checkpoint-11070', 'checkpoint-13284', 'checkpoint-15498', 'checkpoint-17712', 
#              'checkpoint-19926', 'checkpoint-22140', 'checkpoint-24354', 'checkpoint-26568' ]
location = "./saved_models/gen_filter_token3_ddup/"
list_ckp = [ 'checkpoint-1058', 'checkpoint-2116', 'checkpoint-3174', 'checkpoint-4232', 'checkpoint-5290', 'checkpoint-6348', 'checkpoint-7406', 'checkpoint-8464', 
             'checkpoint-9522', 'checkpoint-10580', 'checkpoint-11638', 'checkpoint-12696' ]
last_ckp = ''

order_list = []
for model in list_ckp:
    order_list.append( location + model )
order_list.append( location + last_ckp )

metric = combine( [ 'exact_match', 'rouge' ] )

eval_df = pd.read_pickle( raw_path )
labels = eval_df[ 'target' ]
inputs = eval_df[ 'source' ].tolist()

tokenizer = AutoTokenizer.from_pretrained(tk_checkpoint, trust_remote_code=True)

results = []
scores = []

for model_loc in order_list:
    print( f"Start evaluate model {model_loc}")
    model = AutoModel.from_pretrained( model_loc, trust_remote_code=True, ignore_mismatched_sizes=True ).to( device )

    generated_texts = []

    model.eval()

    batch_size = 100
    input_size = len( inputs )
    iter_count = int( input_size / batch_size ) + 1

    for i in range( iter_count ):
        print( i )
        if( ( i + 1 ) * batch_size < input_size ):
            tmp_input_ids = inputs[ i * batch_size : ( i + 1 ) * batch_size ]
        else:
            tmp_input_ids = inputs[ i * batch_size : input_size ]
        input_ids = tokenizer( tmp_input_ids, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=tokenizer.model_max_length ).input_ids.to( device )

        with torch.no_grad():
            generated_ids = model.generate( input_ids, max_length=128 )
        
        generated_texts.extend( tokenizer.batch_decode( generated_ids, skip_special_tokens=True ) )
    
    results.append( generated_texts )

    score = metric.compute( predictions=generated_texts, references=labels )

    scores.append( score )
    print( '-' * 10 )
    
df = pd.DataFrame()
df['generated_texts'] = results
df['scores'] = scores

df.to_pickle( './token3_results.pkl' )

# print( result )

# tokenizer = AutoTokenizer.from_pretrained(tk_checkpoint, trust_remote_code=True)
# model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, ignore_mismatched_sizes=True).to(device)

# cpp_tokenizer = CppTokenizer()

# #exact_match_metric = load( 'exact_match' )
# metric = combine( [ 'exact_match', 'rouge' ] )

# eval_df = pd.read_pickle( raw_path )
# eval_df = eval_df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True )
# print( eval_df )
# labels = eval_df['target']
# inputs = eval_df['source'].tolist()
# input_ids = tokenizer( inputs, return_tensors='pt', padding='max_length', truncation=True,
#                         max_length=tokenizer.model_max_length ).input_ids.to( device )
# generated_texts = []

# # print( eval_data )
# # print( input_ids )

# model.eval()

# batch_size = 100
# input_size = len( input_ids )
# iter_count = int( input_size / batch_size ) + 1

# for i in range( iter_count ):

#     if( ( i + 1 ) * batch_size < input_size ):
#         tmp_input_ids = inputs[ i * batch_size : ( i + 1 ) * batch_size ]
#         input_ids = tokenizer( tmp_input_ids, return_tensors='pt', padding='max_length', truncation=True,
#                         max_length=tokenizer.model_max_length ).input_ids.to( device )
#     else:
#         tmp_input_ids = inputs[ i * batch_size : input_size ]
#         input_ids = tokenizer( tmp_input_ids, return_tensors='pt', padding='max_length', truncation=True,
#                         max_length=tokenizer.model_max_length ).input_ids.to( device )
    
#     with torch.no_grad():
#         generated_ids = model.generate( input_ids, max_length=128 )
    
#     generated_texts.extend( tokenizer.batch_decode( generated_ids, skip_special_tokens=True ) )

# # print( generated_texts )

# results_output = 'results/b8g1.pkl' 

# df = pd.DataFrame( generated_texts )
# df['labels'] = labels
# df.to_pickle( results_output )

# print( len( labels ), len( generated_texts ) )

# # exact match: 0, 1
# #em = exact_match_metric.compute( predictions=generated_texts, references=labels )
# # partial match: percentage of match, raw result, blu score, CodeBLEU
# #pm = []

# #for i in range( input_size ):
# #
# #    tokens_label = [ token.token_value for token in cpp_tokenizer.tokenize( labels[i] ) ]
# #    tokens_gen = [ token.token_value for token in cpp_tokenizer.tokenize( generated_texts[i] ) ]
# #
# #    raw_result = []
# #
# #    for token in tokens_label:
# #        if token in tokens_gen:
# #            raw_result.append(1)
# #        else:
# #            raw_result.append(0)
# #
# #    percent = sum( raw_result ) / len( raw_result )
# #    
# #    pm.append({
# #        'percentage': percent,
# #        'raw_result': raw_result,
# #        'label_len': len( tokens_label ),
# #        'gen_len': len( tokens_gen )
# #    })
# #
# #pm_df = pd.DataFrame( pm )

# result = metric.compute( predictions = generated_texts, references = labels )

# print( result )

#print( f"Evaluation size: {len(labels)}" )
#print( '-' * 10 )
#print( f"Exact Match: {em['exact_match']}" )
#@print( '-' * 10 )
#@print( pm_df[ pm_df['percentage'] < 1 ])
#@print( f"Partial Match: {len(pm_df[ pm_df['percentage'] < 1 ])} / {len(pm_df)}")
#@print( f"Match average: {pm_df['percentage'].mean()}")
