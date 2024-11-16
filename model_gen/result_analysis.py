import pandas as pd

ori_df = pd.read_pickle( '../data/eval_data_filter_token2_ddup.pkl' )
res_df = pd.read_pickle( 'noe_results.pkl' )

ori_list = ori_df['target'].tolist()
res_list = res_df.iloc[12]['generated_texts']

# print( len( ori_list), len( res_list )) 

results = []

for i in range( 0, len(res_list) ):
    if( res_list[i] == ori_list[i] ):
        results.append( res_list[i] )

df = pd.DataFrame( results )

print( len( df ) )
print( df.value_counts( normalize=True ) )