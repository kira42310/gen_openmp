import pandas as pd

def clean_space( data ):
    # return regex.sub( '', ' '.join( raw[ int( data['start_line'] ) - 1: int( data['end_line'] ) ] ) )
    return ' '.join( data.split() )

tr_df = pd.read_pickle( '../data/training_data_filter_token2.pkl' )
# ev_df = pd.read_pickle( '../data/eval_data_filter_token2_ddup.pkl' )

# tr_df = pd.read_pickle( './training_data_filter_token2_ddup.pkl' )
# ev_df = pd.read_pickle( './eval_data_filter_token2_ddup2.pkl' )

# ft_df = pd.read_pickle( '../data/ft.pkl' )
# hpccg_df = pd.read_pickle( '../data/hpccg.pkl' )

# print( tr_df )
# print( '-*-' * 20 )
# print( ev_df )
# print( '-*-' * 20 )
# print( tr_df['target'].value_counts( normalize=True ) )
# print( '-*-' * 20 )
# print( ev_df['target'].value_counts( normalize=True ) )

# ft_df = pd.read_pickle( '../data/ft.pkl' )
# hpccg_df = pd.read_pickle( '../data/hpccg.pkl' )

# tr_list = tr_df['source'].tolist()
# # ev_list = ev_df['source'].tolist()
# ft_list = ft_df['source'].tolist()
# hpccg_list = hpccg_df['source'].tolist()

# print( len( ft_list ), len( hpccg_list ) )

# check_ = ft_list + hpccg_list
# check_list = []
# for i in check_:
#     check_list.append( clean_space( i ) )

# print( check_list, len( check_list ) )

# dup = []

# for i, ev in enumerate( ev_list ):
#     if( ev in tr_list ):
#         dup.append( i )

# for i, ev in enumerate( check_list ):
#     if( ev in tr_list ):
#         dup.append( { 'i': i, 'dup': True } )
#     else:
#         dup.append( { 'i': i, 'dup': False } )
# print( dup )

# print( len(dup) )

# new_ev_df = ev_df.drop( dup ).reset_index( drop=True )

# print( new_ev_df )

# new_ev_df.to_pickle( 'eval_data_filter_token2_ddup2.pkl' )

# for i, d in enumerate( tr_list ):
#     if d == hpccg_list[5]:
#         print( i )
#         break
#         # ans == 65112
# print( tr_df.iloc[65112] )
# print( tr_df.iloc[65112]['source'] )

tr_df['count'] = tr_df.groupby('source')['source'].transform( 'count' )
dup = tr_df.sort_values( ['count','source'] , ascending=[False,True] )
print( dup.iloc[:84]['repo_name'].tolist() )