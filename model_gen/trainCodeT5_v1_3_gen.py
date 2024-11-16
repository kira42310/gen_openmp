import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import cuda

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'

# define a rich console logger
console = Console(record=True)

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)



class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train( epoch, tokenizer, model, device, loader, optimizer, accumulation_steps ):
    model.train()
    for _, data in enumerate( loader, 0 ):
        y = data['target_ids'].to( device, dtype=torch.long )
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[ y[:, 1:] == tokenizer.pad_token_id ] = -100
        ids = data[ 'source_ids' ].to( device, dtype=torch.long )
        mask = data[ 'source_mask' ].to( device, dtype=torch.long )
        
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        # loss = outputs[0]
        loss = outputs[0].mean()
        loss = loss / accumulation_steps
        loss.backward()
        
        if( (_+1) % accumulation_steps == 0 or (_+1) == len( loader ) ):
            optimizer.step()
            optimizer.zero_grad()

def validate( epoch, tokenizer, model, device, loader ):
    model.eval()
    predictions = []
    actuals = []
    probs = []
    with torch.no_grad():
        for _, data in enumerate( loader, 0 ):
            y = data[ 'target_ids' ].to( device, dtype=torch.long )
            ids = data[ 'source_ids' ].to( device, dtype=torch.long )
            mask = data[ 'source_mask' ].to( device, dtype=torch.long )
            
            generated_ids = model.module.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            preds = [ tokenizer.decode( g, skip_special_tokens=True, clean_up_tokenization_spaces=True ) for g in generated_ids.sequences ]
            target = [ tokenizer.decode( t, skip_special_tokens=True, clean_up_tokenization_spaces=True ) for t in y ]
            scores = model.module.compute_transition_scores( generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices, normalize_logits=False)
            
            output_length = 1 + np.sum(scores.to('cpu').numpy() < 0, axis=1)
            length_penalty = model.module.generation_config.length_penalty
            reconstructed_scores = scores.to('cpu').sum(axis=1) / (output_length**length_penalty)   
            
            #if _ % 10 == 0:
            #    console.print( f'Completed {_}' )
                
            predictions.extend( preds )
            actuals.extend( target )
            probs.extend( reconstructed_scores )
        
        probs = np.exp( probs )
        
        return predictions, actuals, probs

def T5Trainer( dataframe_train, dataframe_valid, source_text, target_text, model_params, output_dir="./outputs" ):
    torch.manual_seed( model_params[ 'SEED' ] )
    np.random.seed( model_params[ 'SEED' ] )
    torch.backends.cudnn.deterministic = True
    
    console.log( f'''[model]: Loading {model_params['MODEL']}...''' )
    
    tokenizer = RobertaTokenizer.from_pretrained( model_params['MODEL'] )
    
    model = T5ForConditionalGeneration.from_pretrained( model_params['MODEL'] )
    if( device == 'cuda' ):
        model = torch.nn.DataParallel( model )
    model = model.to( device )
    
    console.log( f'[DATA]: Reading data... \n' )

    train_dataset = dataframe_train.sample( frac=1, random_state=model_params['SEED'] ).reset_index( drop=True )
    val_dataset = dataframe_valid
    
    training_set = YourDataSetClass( train_dataset, tokenizer, model_params['MAX_SOURCE_TEXT_LENGTH'], model_params['MAX_TARGET_TEXT_LENGTH'], source_text, target_text )
    val_set = YourDataSetClass( val_dataset, tokenizer, model_params['MAX_SOURCE_TEXT_LENGTH'], model_params['MAX_TARGET_TEXT_LENGTH'], source_text, target_text )
    
    train_params = {
        'batch_size': model_params['TRAIN_BATCH_SIZE'],
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True
    }
    val_params = {
        'batch_size': model_params['VALID_BATCH_SIZE'],
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True
    }
    
    training_loader = DataLoader( training_set, **train_params )
    val_loader = DataLoader( val_set, **val_params )
    
    optimizer = torch.optim.Adam( params=model.parameters(), lr=model_params[ 'LEARNING_RATE' ] )
    
    console.log( f'[Initiating fine tuning]...\n' )
            
    max_f1 = 0
    previous_f1 = 0
    early_stop = model_params['EARLY_STOP']
    for epoch in range( model_params[ 'TRAIN_EPOCHS' ] ):
        train( epoch, tokenizer, model, device, training_loader, optimizer, model_params['GRADIENT_ACCUMULATION'] )
        
        predictions, actuals, probs = validate( epoch, tokenizer, model, device, val_loader )
        pos_prob = []
        for i, d in enumerate( predictions ):
            if d == '0':
                pos_prob.append( 1 - probs[i] )
            else:
                pos_prob.append( probs[i] )

        df = pd.DataFrame( {'Generated Text': predictions, 'Actual Text': actuals} )
        df.to_pickle( os.path.join( output_dir, 'pred_' + str(epoch + 1) + '.pkl' ) )
       
        #predictions = [ str(int(i)) if i != '' else '0' for i in predictions ]
        #actuals = [ str(int(i)) if i != '' else '0' for i in actuals ]
        
        prec, recall, threh = precision_recall_curve( actuals, pos_prob, pos_label='1' )
        
        #df = pd.DataFrame( {'Generated Text': predictions, 'Actual Text': actuals} )
        #df['Perfect Prediction'] = df['Generated Text'] == df['Actual Text']
        #if( (epoch + 1) % 5 == 0 ):
        #    df.to_csv( os.path.join( output_dir, 'pred_' + str(epoch + 1) + '.csv' ) )
        #df.to_csv( os.path.join( output_dir, 'pred_' + str(epoch + 1) + '.csv' ) )
       
        #ppr = ( df['Perfect Prediction'] == True ).sum()/len(df)
        acc = accuracy_score( actuals, predictions )
        f1 = f1_score( actuals, predictions, average='binary', pos_label='1' )
        prec_s = precision_score( actuals, predictions, average='binary', pos_label='1' )
        recall_s = recall_score( actuals, predictions, average='binary', pos_label='1' )
        console.print(f'epoch: {epoch+1}, loss: , results: { acc }, f1: {f1}, prec: {prec_s}, recall: {recall_s}, PR_AUC: {auc( recall, prec )}, ROC_AUC: {roc_auc_score( actuals, pos_prob )}')
        if( f1 > max_f1 ):
            console.print( f'### New highest f1 epoch: {epoch+1} ###' )
            max_f1 = f1
            path = os.path.join( output_dir, 'highest_model' )
            model.module.save_pretrained( path )
            tokenizer.save_pretrained( path )
            #df.to_csv( os.path.join( path, 'highest_model_pred.csv' ) )
        
        if( f1 <= previous_f1 ):
            early_stop -= 1
            if( early_stop <= 0 ):
                console.print( f'### early stop @ {epoch + 1} ###' )
                break
        else:
            previous_f1 = f1
            early_stop = model_params['EARLY_STOP']
    
    console.log( f'[Saving Model]...\n' )
    
    path = os.path.join( output_dir, 'model_files' )
    #model.save_pretrained( path )
    model.module.save_pretrained( path )
    tokenizer.save_pretrained( path )
    
    console.save_text( os.path.join( output_dir, 'logs.txt' ) )
    
    console.log( f'[Validation completed.]\n' )
    console.print( f'''[Model] Model saved @ { os.path.join( output_dir, 'model_files' ) } \n''' )
    console.print( f'''[Validation] Generation on validation data saved @ { os.path.join( output_dir, 'predictions.csv' ) }\n''' )
    console.print( f'''[Logs] Logs saved @ { os.path.join( output_dir, 'logs.txt' ) }\n''' )

model_name = "Salesforce/codet5-small"
batch = int(sys.argv[3])
grad_accu = int(sys.argv[4])
early_stop = 8
epoch = int(sys.argv[2])
#lr = 2e-4
lr = float(sys.argv[5])
output_dir = sys.argv[1]

model_params = {
    "MODEL": model_name,  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": int(batch/grad_accu),  # training batch size
    "VALID_BATCH_SIZE": int(batch/grad_accu),  # validation batch size
    "TRAIN_EPOCHS": epoch,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": lr,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 4,  # max length of target text
    "SEED": 4,  # set seed for reproducibility
    "EPOCH_CHECK_POINT": 5, # checkpint
    "GRADIENT_ACCUMULATION": grad_accu,
    "EARLY_STOP": early_stop,
}

trainlist = pd.read_pickle('/work/soratouch-p/project/codet5/data/gn_model_train.pkl')
validlist = pd.read_pickle('/work/soratouch-p/project/codet5/data/gn_model_valid.pkl')

T5Trainer(
    dataframe_train=trainlist,
    dataframe_valid=validlist,
    source_text='source',
    target_text='target',
    model_params=model_params,
    output_dir=output_dir
)
