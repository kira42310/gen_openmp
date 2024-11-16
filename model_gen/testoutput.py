from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch

tk_checkpoint = "Salesforce/codet5p-220m-bimodal"
# checkpoint = "./test/final_checkpoint"
checkpoint = "./saved_models/gen_filter_token_noe/"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(tk_checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, ignore_mismatched_sizes=True).to(device)

def load_tokenize_data( raw_path, cache_path ):
    if os.path.exists( cache_path ):
        eval_data = load_from_disk( cache_path )
        print( f' ==> {len(eval_data)} samples')
        return eval_data
    else:
        ev_df = pd.read_pickle( raw_path )
        evsets = Dataset.from_pandas( ev_df )

        def preprocess_function( examples ):
            source = examples['source']
            target = examples['target']

            model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            return model_inputs
        
        eval_data = evsets.map(
           preprocess_function,
           batched=True,
           remove_columns=datasets.column_names,
           num_proc=2,
           load_from_cache_file=False,
        )
        print(f'  ==> Saved to {args.cache_data}')
        eval_data.save_to_disk( args.eval_data )
        return eval_data

def compute_metrics( preds, labels ):

    print( '' )
    print( len( preds ) )
    print( preds[0] )
    print( len( labels ) )
    print( labels[0] )

    # decoded_preds = [ tokenizer.decode( pred, skip_special_tokens=True ) for pred in preds ]
    # decoded_labels = [ tokenizer.decode( label, skip_special_tokens=True ) for label in labels ]
    print( '-----------------------------1-----------------------------' )
    # decoded_preds = tokenizer.batch_decode( preds, skip_special_tokens=True )
    decoded_preds = [ tokenizer.decode( pred, skip_special_tokens=True ) for pred in preds ]
    print( '-----------------------------2-----------------------------' )
    # labels = [ [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"] ]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    print( '-----------------------------3-----------------------------' )
    decoded_labels = tokenizer.batch_decode( labels, skip_special_tokens=True )
    print( '-----------------------------4-----------------------------' )
    # predictions = np.
    # labels = pred.label_ids
    # preds = pred.prediction.argmax(-1)

    # em = exact match
    # em = sum( [ 1 if p == l else 0 for p, l in zip( preds, labels ) ]  ) / len( labels )

    # f1 = f1_score( labels, preds, average='marco' )
    
    # return {
    #     'f1': f1,
    #     'exact_match': em
    # }
    print( decoded_labels )
    # print( labels )
    print( decoded_preds )
    results = metric.compute( predictions=decoded_preds, references=decoded_labels )
    #results = metric.compute( predictions=decoded_preds, references=labels )
    # results = metric.compute( predictions=preds, references=labels )
    #print( results )
    return results

# raw_path = '../data/eval_data_filter_token2.pkl'

# eval_df = pd.read_pickle( raw_path )
# eval_raw = eval_df['source'].values.tolist()
# label_raw = eval_df['target'].values.tolist()

c1 = """for( int i = 0; i < 100 < i++ )
{
    c[i] = a[i] + b[i]
}"""

mm = """for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
        rslt[i][j] = 0;
 
        for (int k = 0; k < R2; k++) {
            rslt[i][j] += mat1[i][k] * mat2[k][j];
        }
 
        cout << rslt[i][j] << "\t";
    }
 
    cout << endl;
}"""

mm2 = """for( int i = 0; i < N; i++ ){
    for( int j = 0; j < N; j++ ){
        for( int k = 0; k < N; k++ )
            C[i][j] += A[i][k] * B[k][j];
    }
}
"""

fftw = """for (i = 0; i < nthr; ++i) {
	  d.max = (d.min = i * block_size) + block_size;
	  if (d.max > loopmax)
	       d.max = loopmax;
	  d.thr_num = i;
	  d.data = data;
	  proc(&d);
     }"""

stencil = """			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a1[i*size*size+j*size+k] = (
								a0[i*size*size+(j-1)*size+k] + a0[i*size*size+(j+1)*size+k] +
								a0[(i-1)*size*size+j*size+k] + a0[(i+1)*size*size+j*size+k] +
								a0[(i-1)*size*size+(j-1)*size+k] + a0[(i-1)*size*size+(j+1)*size+k] +
								a0[(i+1)*size*size+(j-1)*size+k] + a0[(i+1)*size*size+(j+1)*size+k] +

								a0[i*size*size+(j-1)*size+(k-1)] + a0[i*size*size+(j+1)*size+(k-1)] +
								a0[(i-1)*size*size+j*size+(k-1)] + a0[(i+1)*size*size+j*size+(k-1)] +
								a0[(i-1)*size*size+(j-1)*size+(k-1)] + a0[(i-1)*size*size+(j+1)*size+(k-1)] +
								a0[(i+1)*size*size+(j-1)*size+(k-1)] + a0[(i+1)*size*size+(j+1)*size+(k-1)] +

								a0[i*size*size+(j-1)*size+(k+1)] + a0[i*size*size+(j+1)*size+(k+1)] +
								a0[(i-1)*size*size+j*size+(k+1)] + a0[(i+1)*size*size+j*size+(k+1)] +
								a0[(i-1)*size*size+(j-1)*size+(k+1)] + a0[(i-1)*size*size+(j+1)*size+(k+1)] +
								a0[(i+1)*size*size+(j-1)*size+(k+1)] + a0[(i+1)*size*size+(j+1)*size+(k+1)] +

								a0[i*size*size+j*size+(k-1)] + a0[i*size*size+j*size+(k+1)]
						) * fac;
					}
				}
			}"""

stencil2 = """			for (i = 1; i < n+1; i++) {
				for (j = 1; j < n+1; j++) {
					for (k = 1; k < n+1; k++) {
						a0[i*size*size+j*size+k] = a1[i*size*size+j*size+k];
					}
				}
			}
		}"""

code = mm2

# i = 300

# eval_raw = eval_raw[i]

# input_ids = tokenizer( eval_raw, return_tensors='pt', padding='max_length',
#                         truncation=True, max_length=tokenizer.model_max_length).input_ids.to( device )

# with torch.no_grad():
#     generated_ids = model.generate( input_ids, max_length=128 )

# generated_texts = tokenizer.batch_decode( generated_ids, skip_special_tokens=True )
# print( eval_raw )
# print( label_raw[i] )
# print( generated_texts[0] )

# code = """def svg_to_image(string, size=None):
#     if isinstance(string, unicode):
#         string = string.encode('utf-8')
#         renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
#     if not renderer.isValid():
#         raise ValueError('Invalid SVG data.')
#     if size is None:
#         size = renderer.defaultSize()
#         image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
#         painter = QtGui.QPainter(image)
#         renderer.render(painter)
#     return image"""

# code = """for( int i = 0; i < 100 < i++ )
# {
#     c[i] = a[i] + b[i]
# }"""

input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(input_ids, max_length=20)
print( code )
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# Convert a string of SVG data to an image.
