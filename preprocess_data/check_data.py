import pandas as pd

df = pd.read_pickle( '../data/training_data_filter_token3.pkl' )
edf = pd.read_pickle( '../data/eval_data_filter_token3.pkl' )

print( df )
print( edf )
# print( df[['for_raw', 'omp_raw', 'tokens_for', 'tokens_omp']])
# print( df[ df['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ] ) 
# print( edf[ edf['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ] ) 