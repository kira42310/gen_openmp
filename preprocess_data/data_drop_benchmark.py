import pandas as pd

tr_df = pd.read_pickle( '../data/training_data_filter_token2.pkl')
# ori_df = pd.read_pickle( '../data/data202312_stack_v5.pkl' )
npb_df = pd.read_pickle( '../data/npb.pkl' )
print( tr_df )
print( npb_df.columns )

# res_df = ori_df[ (ori_df['repo_name'] == 'GMAP/NPB-CPP') & (ori_df['parallel'] == 1) ] 
# print( res_df )
tr_df = tr_df[ (tr_df['repo_name'] != 'GMAP/NPB-CPP') ] 
print( tr_df )

# res_df.to_pickle( '../data/npb.pkl' )

npb_list = npb_df['for_raw'].tolist()

drop_index = []
for i, d in tr_df.iterrows():
    if d['source'] in npb_list:
        if( i not in drop_index ):
            drop_index.append( i )

print( drop_index )

tr_df = tr_df.drop( drop_index )

print( tr_df )

tr_df = tr_df.reset_index( drop=True )

print( tr_df )

tr_df.to_pickle( '../data/training_data_filter_token2_dnpb.pkl' )

tr_df2 = tr_df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True )

print( tr_df2 )

tr_df2.to_pickle( '../data/training_data_filter_token2_ddup_dnpb.pkl' )