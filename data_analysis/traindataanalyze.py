import pandas as pd

tr_df = pd.read_pickle( '../data/training_data_filter_token2.pkl' )

print( tr_df[['source','target']].duplicated().value_counts() )

print( tr_df['source'].duplicated().value_counts() )

res = tr_df['source'].value_counts(  )

print( res[:10] )

new_tr_df = tr_df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True )

print( new_tr_df )

new_tr_df.to_pickle( 'training_data_filter_token2_ddup.pkl' )