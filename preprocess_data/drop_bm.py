import pandas as pd

df = pd.read_pickle( '../data/training_data_filter_token3.pkl' )
edf = pd.read_pickle( '../data/eval_data_filter_token3.pkl' )

print( df )
print( edf )

print( df.drop( df[ df['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ].index.values.tolist() ) )
print( edf.drop( edf[ edf['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ].index.values.tolist() ) )

df = df.drop( df[ df['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ].index.values.tolist() ).reset_index( drop=True )
edf = edf.drop( edf[ edf['repo_name'].isin( ['GMAP/NPB-CPP', 'cavazos-lab/PolyBench-ACC'] ) ].index.values.tolist() ).reset_index( drop=True )

df.to_pickle( '../data/training_data_filter_token3.pkl' )
edf.to_pickle( '../data/eval_data_filter_token3.pkl' )

print( df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True ) )
print( edf.drop_duplicates( subset=['source','target'] ).reset_index( drop=True ) )

df = df.drop_duplicates( subset=['source','target'] ).reset_index( drop=True ).reset_index( drop=True )
edf = edf.drop_duplicates( subset=['source','target'] ).reset_index( drop=True ).reset_index( drop=True )

df.to_pickle( '../data/training_data_filter_token3_ddup.pkl' )
edf.to_pickle( '../data/eval_data_filter_token3_ddup.pkl' )