import pandas as pd
import preprocess_util

base_location = '/home/mudang/script/npb-prep/'
filter_location = base_location + 'filter/'
bms = ['bt','cg','ep','ft','is','lu','mg','sp']

def fix_space_nline( data ):
    fix_data = []
    for d in data:
        fix_data.append( preprocess_util.fix_space_nline( d ) )
    return fix_data

combind = pd.DataFrame()
combind_withdata = pd.DataFrame()
for bm in bms:
    bm = pd.read_pickle( filter_location + bm + '.pkl' ).reset_index(drop=True)
    for_raw_list = fix_space_nline( bm['for_raw'] )
    bm = bm.drop( columns=['for_raw'] )
    bm.insert( len( bm.columns ), 'for_raw', for_raw_list )
    combind_withdata = pd.concat( [combind_withdata, bm] ).reset_index(drop=True)
    combind = pd.concat( [combind, pd.DataFrame( for_raw_list, columns=['source'])]).reset_index( drop=True )
    
print( combind_withdata )
print( combind )

combind.to_pickle( filter_location + 'npb_eval.pkl' )
combind_withdata.to_pickle( filter_location + 'combind_bm.pkl' )