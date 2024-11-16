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

raw_path = '../data/hpccg.pkl'
# raw_path = '../data/ft.pkl'
cache_path = './cache_data/eval_cache/'

location = "../model_gen/saved_models/gen_filter_token_noe_dnpb/"

ft_df = pd.read_pickle( raw_path )

inputs = ft_df[ 'source' ].tolist()

model = AutoModel.from_pretrained( location, trust_remote_code=True, ignore_mismatched_sizes=True ).to( device )
tokenizer = AutoTokenizer.from_pretrained(tk_checkpoint, trust_remote_code=True)

gen_text = []
for i in inputs:
    input_ids = tokenizer( i, return_tensors="pt" ).input_ids.to( device )
    
    generated_ids = model.generate( input_ids, max_length=128 )

    gen_text.append( tokenizer.decode( generated_ids[0], skip_special_tokens=True ) )

for i in gen_text:
    print( i )

df = pd.DataFrame( gen_text )
df.to_csv( "output.txt", sep='\n', index=False )