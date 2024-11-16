import pandas as pd
import json
import os
import preprocess_util
import shutil

from collections import deque

npb_location = '/home/mudang/script/npb-prep/NPB-OMP/'

npb_files = [
    'BT/bt.cpp',
    'CG/cg.cpp',
    'EP/ep.cpp',
    'FT/ft.cpp',
    'IS/is.cpp',
    'LU/lu.cpp',
    'MG/mg.cpp',
    'SP/sp.cpp'
]

tmp_cc = '/data/8t/tmpcc.cpp'
tmp_output = '/data/8t/tmp'
output_location = '/home/mudang/script/npb-prep/ast/'
home_location = '/home/mudang/script/npb-prep/'

for bm in npb_files:
    if( os.path.exists( tmp_cc ) ):
        os.remove( tmp_cc )

    if( os.path.exists( tmp_output ) ):
        os.remove( tmp_output )
    
    preprocess_util.clean_comment_c( npb_location + bm, tmp_cc, timeout=60 )
    shutil.copyfile( tmp_cc, os.path.join( home_location, 'raw-cc/'+ bm.split('/')[1] +'.omp' ) )
    preprocess_util.ast_dump_json( tmp_cc, tmp_output, timeout=60 )
    
    with open( tmp_output, 'r' ) as j_f:
        json_data = json.load( j_f )
    ast_stack = deque( json_data['inner'] )
    del json_data
    if( os.path.exists( tmp_output ) ):
        os.remove( tmp_output )
    ast = preprocess_util.ast_traverse_stack( ast_stack )
    # tmp_data = {
    #     'bm': bm,
    #     'data': ast
    # }
    with open( tmp_cc ) as f:
        raw_file = f.readlines()
    data_list = []
    for i in ast:
        data_list.extend( preprocess_util.label_data( i, raw_file, bm, '-' ) )
    # try:
    #     tmp_data = preprocess_util.label_data( ast, raw_file, bm, '-' )
    # except:
    #     print( len(ast) )
    pd.DataFrame( data_list ).to_pickle( output_location + bm.split('/')[1] + 'omp.pkl' )