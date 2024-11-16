import pandas as pd

import preprocess_util

def drop_dup( df ):
    return df.sort_values( [ 'for_start', 'for_end', 'parallel' ], ascending=[ True, True, False ] ) \
            .duplicated( subset=['for_start', 'for_end'] ) \

df = pd.read_pickle( '../data/data202312_stack_v5.pkl' )

df = preprocess_util.create_training_data_generative( df )

# df = df[df['parallel'] == 1]

# stars = []
# repo_name = []
# tmp = df['file_path']
# for path in tmp:
#     p = path.split('/')
#     stars.append( p[2] )
#     repo_name.append( '/'.join( [ p[3],p[4] ] ) )

# df['stars'] = stars
# df['repo_name'] = repo_name

# stars_count = df['stars'].value_counts()
# stars_count = stars_count / len( df ) * 100
# dup_count = df.duplicated(subset=['for_raw','omp_raw']).value_counts()

#print( stars )

# df_t = pd.read_pickle( '../data/training_data.pkl' )
# df_e = pd.read_pickle( '../data/eval_data.pkl' )

# df = preprocess_util.create_training_data_generative( df, 0.80, 4 )
#df = df[ ~df['omp_raw'].isnull() ]
#df = df.drop_duplicates( subset=['for_raw','omp_raw'] )

# df = df.drop_duplicates( subset=['tokens_for'] )

print( df )
# print( len( df[df['tokens_count_for'] > 20 ] ) )
# print( df['eval_data'].iloc[100] )
#print( df.columns )
#print( df['file_path'] )
# print( df['stars'].value_counts() )
# print( stars_count )
# for i in df['stars'].unique():
#     print( f'STAR: {i}' )
#     data_df = df[df['stars'] == i ]
#     size = len( data_df )
#     dup_df = data_df.duplicated(subset=['for_raw','omp_raw']).value_counts().sort_index()
#     print( dup_df )
#     print( dup_df[1] / size * 100 )

# dup_df = df.duplicated( subset=['for_raw','omp_raw']).value_counts().sort_index()
# print( dup_df )
# print( dup_df[1] / len(df) * 100 )
# print( df['train_data'] == df_t )
# print( df['eval_data'] == df_e )
#print( df['kind'].value_counts() )

# tmp_df = df[ df['repo_name'] == 'h2o/h2o' ]
# for_dup = tmp_df.duplicated( subset=['for_raw'] ).value_counts().sort_index()
# print( for_dup )
# print( for_dup.index.tolist() )

# repo_stat = []
# for name in df['repo_name'].unique():
#     tmp_df = df[df['repo_name'] == name ]
#     stars = tmp_df.iloc[0]['stars']
#     tmp_dup = tmp_df.duplicated( subset=['for_raw', 'omp_raw'] ).value_counts().sort_index()
#     for_dup = tmp_df.duplicated( subset=['for_raw'] ).value_counts().sort_index()
#     for_dup_count = ( ( 0 if for_dup.index.tolist()[0] == False else for_dup[0] ) if len( for_dup ) == 1 else for_dup[1] )
#     if( len(tmp_dup) == 1 ):
#         if( tmp_dup.index.tolist()[0] == False ):
#             repo_stat.append({
#                 'repo_name': name,
#                 'stars': stars,
#                 'loop': len(tmp_df),
#                 'unique': tmp_dup[0],
#                 'dup': 0,
#                 'for_dup': for_dup_count
#             })
#         else:
#             repo_stat.append({
#                 'repo_name': name,
#                 'stars': stars,
#                 'loop': len(tmp_df),
#                 'unique': 0,
#                 'dup': tmp_dup[0],
#                 'for_dup': for_dup_count
#             })
#     else:
#         repo_stat.append({
#             'repo_name': name,
#             'stars': stars,
#             'loop': len(tmp_df),
#             'unique': tmp_dup[0],
#             'dup': tmp_dup[1],
#             'for_dup': for_dup_count
#         })

# print( '-------------------------------------' )
# repo_stat_df = pd.DataFrame( repo_stat )
# print( repo_stat_df.sort_values(by=['loop','unique','dup']) )
# print( f"AVG: {repo_stat_df['loop'].mean()}" )
# print( f"AVG UNIQUE: {repo_stat_df['unique'].mean()}" )
# print( f"SUM: {repo_stat_df['loop'].sum()}" )
# print( f"SUM unique: {repo_stat_df['unique'].sum()}" )
# print( f"SUM dup: {repo_stat_df['dup'].sum()}" )
# print( f"SUM for dup: {repo_stat_df['for_dup'].sum()}" )
# for i in repo_stat_df['stars'].unique():
#     print( f"{'-'*10} STAR: {i} {'-'*10}" )
#     data_df = repo_stat_df[ repo_stat_df['stars'] == i ]
#     print( data_df.sort_values( by=['loop','unique','dup']) )
#     print( f"SUM: {data_df['loop'].sum()}" )
#     print( f"SUM unique: {data_df['unique'].sum()}" )
#     print( f"SUM dup: {data_df['dup'].sum()}" )
#     print( f"AVG: {data_df['loop'].mean()}" )
#     print( f"AVG UNIQUE: {data_df['unique'].mean()}" )
#     print( f"MAX: {data_df['loop'].max()}" )
#     print( f"MAX UNIQUE: {data_df['unique'].max()}" )
#     print( f"For dup: {data_df['for_dup'].sum()}" )
