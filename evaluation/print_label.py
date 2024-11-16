import pandas as pd

df = pd.read_pickle( '../data/npb.pkl' )

# res = df['file_path']
res = df[ df['file_path'] == '/repos_data/2/GMAP/NPB-CPP/NPB-OMP/FT/ft.cpp' ]['omp_raw'].tolist()

# print( res )
for i in res:
    print( i )