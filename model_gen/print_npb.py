import pandas as pd

df = pd.read_pickle( '../../npb-prep/filter/combind_bm.pkl' )

df = df[['for_start', 'for_end', 'file_path']]

for i, d in df.iterrows():
    print( f"{i+2} {d['file_path']} {d['for_start']} {d['for_end']} \n" )
