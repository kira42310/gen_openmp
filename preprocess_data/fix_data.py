import pandas as pd
import preprocess_util

df = pd.read_pickle( '../data/data202310_stack_v2.pkl' )
df5 = pd.read_pickle( '../data/data202312_stack_v5.pkl' )
# df = pd.read_pickle( '../data/eval_data_filter_token2.pkl' )
# df = pd.read_pickle( '../data/training_data_filter_token2.pkl' )

# df = preprocess_util.add_tokens_to_pd( df )
# df = df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True )
# df = df[:800]

# print( df )
# print( df.columns )
# print( df[['for_raw']])
# print( df.columns )
# print( df.iloc[0] )

# print( len(df) )
# print( df.columns )
# print( len(df5) )
# print( df5.columns )
# print( df[['file_path']] == df5[['file_path']])

# for i in range( len( df )):
#     if df.iloc[i]['file_path'] != df5.iloc[i]['file_path']:
#         print( i )
#         break

raw_source = df[['for_raw', 'omp_raw']]
for_raw_list = []
omp_raw_list = []
for i, d in raw_source.iterrows():
    for_tmp = d['for_raw']
    omp_tmp = d['omp_raw']

    #### fix raw ####
    for_raw = []
    for line in for_tmp.split( '\n' ):
        tmp = ' '.join( line.split() )
        if( not tmp.startswith('#pragma omp') ):
            for_raw.append( tmp )
    for_raw_list.append( ' '.join( for_raw ) )

    #### fix omp ####
    if( omp_tmp == None ):
        omp_raw_list.append( None )
    else:
        omp_tmp = preprocess_util.fix_space_nline( omp_tmp )
        omp_tmp = omp_tmp.split()
        if( 'parallel' not in omp_tmp and 'teams' not in omp_tmp ):
            omp_tmp.insert( 2, 'parallel' )
        tmp = []
        for k in omp_tmp:
            if( k != 'nowait' and k != '\\' ):
                tmp.append( k )
        # print( k )
        omp_raw_list.append( ' '.join( tmp ) )

df5 = df5.drop( columns=['for_raw', 'omp_raw', 'tokens_for', 'tokens_count_for', 'tokens_omp', 'tokens_count_omp' ] )

df5['for_raw'] = for_raw_list
df5['omp_raw'] = omp_raw_list

df5 = preprocess_util.add_tokens_to_pd( df5 )

print( df5 )

df5.to_pickle( '../data/data202404_stack_v6.pkl' )

