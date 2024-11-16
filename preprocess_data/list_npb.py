import pandas as pd

df = pd.read_pickle( '../evaluate_npb_omp_gen.pkl' )

for i, d in df[['for_start', 'file_path', 'omp_gen' ]].iterrows():
    print( f"{d['file_path']} {d['for_start']} {d['omp_gen']}" )