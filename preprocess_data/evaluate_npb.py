import pandas as pd

npb_location = '/home/mudang/script/npb-prep/'

npb_df = pd.read_pickle( npb_location + 'filter/combind_bm.pkl' )

print( npb_df )

output_df = [] 
with open( '../model_gen/output_npb_edit.txt', 'r' ) as f:
    output_tmp = f.readlines()

output_list = []
for omp in output_tmp:
    output_list.append( omp.split('\n')[0] )

print( len( output_list ) )
print( output_list )

npb_df.insert( len( npb_df.columns ), 'omp_gen', output_list )

print( npb_df )

npb_df.to_pickle( '../evaluate_npb_omp_gen.pkl' )