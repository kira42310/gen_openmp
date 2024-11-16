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

# raw_path = '../data/hpccg.pkl'
raw_path = '../../npb-prep/filter/npb_eval.pkl'
cache_path = './cache_data/eval_cache/'

#location = "./saved_models/gen_filter_token_noe_ddup_dnpb/"
location = "./saved_models/gen_filter_token3_ddup/checkpoint-9522/"

# ft_df = pd.read_pickle( raw_path )
# inputs = ft_df[ 'source' ].tolist()

inputs = ['''for(j = 0; j < lastcol - firstcol + 1; j++){ norm_temp1 = norm_temp1 + x[j]*z[j]; norm_temp2 = norm_temp2 + z[j]*z[j]; }''',
'''for(j = 0; j < lastcol - firstcol + 1; j++){ x[j] = norm_temp2 * z[j]; }''',
'''for(i3 = 0;i3 < n3; i3++){ for(i2 = 0; i2 < n2; i2++){ for(i1 = 0; i1 < n1; i1++){ z[i3][i2][i1] = 0.0; } } }''']

model = AutoModel.from_pretrained( location, trust_remote_code=True, ignore_mismatched_sizes=True ).to( device )
tokenizer = AutoTokenizer.from_pretrained(tk_checkpoint, trust_remote_code=True)

gen_text = []
for i in inputs:
    input_ids = tokenizer( i, return_tensors="pt" ).input_ids.to( device )
    
    generated_ids = model.generate( input_ids, max_length=128 )

    gen_text.append( tokenizer.decode( generated_ids[0], skip_special_tokens=True ) )

for i in gen_text:
    print( i )

# df = pd.DataFrame( gen_text )
# for i, d in 
# df.to_csv( "output.txt", sep='\n', index=False )