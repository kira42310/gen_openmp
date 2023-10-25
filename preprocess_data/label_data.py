import os
import sys
import pandas as pd
import re
import pprint

import preprocess_util

from datetime import datetime

# want_omp = [
#     'OMPParallelForDirective',
#     'OMPForDirective',
#     'OMPParallelForSimdDirective',
#     'OMPDistributeParallelForSimdDirective',
#     'OMPForSimdDirective',
#     'OMPSimdDirective',
#     'OMPTargetParallelForDirective',
#     'OMPTargetParallelForSimdDirective',
#     'OMPTargetTeamsDistributeParallelForSimdDirective',
#     'OMPTargetSimdDirective',
#     'OMPDistributeSimdDirective',
#     'OMPTargetTeamsDistributeSimdDirective',
#     'OMPTargetTeamsDistributeParallelForDirective',
#     'OMPTaskLoopDirective',
#     'OMPDistributeParallelForDirective',
#     'OMPTaskLoopSimdDirective',
#     'OMPTargetTeamsDistributeDirective',
#     'OMPTeamsDistributeParallelForSimdDirective',
#     'OMPTeamsDistributeSimdDirective',
#     'OMPTeamsDistributeParallelForDirective',
#     'OMPTeamsDistributeDirective',
#     'OMPMasterTaskLoopSimdDirective',
#     'OMPParallelMasterTaskLoopSimdDirective',
#     'OMPDistributeDirective',
#     'OMPMasterTaskLoopDirective',
#     'OMPParallelMasterTaskLoopDirective',
#     # 'OMPParallelDirective'
# ]

cur_location = '/work/soratouch-p/repos_data'

log_file = cur_location + '/error/label.log'
err_file = cur_location + '/error/label.err'

if __name__ == '__main__':

    data_df = pd.DataFrame()

    file_list = preprocess_util.list_files( cur_location + '/output' )

    for i in file_list:
        try:
            data_list = []
            df = pd.read_pickle( i )
            if( df.empty ):
                with open( log_file, 'a' ) as l_f:
                    l_f.write( f'EMPTY {i}\n' )
                continue
            filename = df.iloc[0]['file_path']
            filename_nc = df.iloc[0]['uncomment_source']
            with open( filename_nc, 'r' ) as s_f:
                raw = s_f.readlines()
            for i, row in df.iterrows():
                data = row['data']
                data = preprocess_util.drop_dup( [data] )
                for d in data:
                    data_list.extend( preprocess_util.label_data( d, raw, filename, filename_nc ) )
            data_df = pd.concat( [ data_df, preprocess_util.drop_dup_pd( pd.DataFrame(data_list) ) ], ignore_index=True )
        
        except:
            with open( err_file, 'a' ) as e_f:
                e_f.write( f'OTHER {i} {filename}\n' )

    # new_df = pd.DataFrame( data_list )

    # print( new_df )
    # pd.DataFrame( data_list ).to_pickle( './data202310.pkl' )
    data_df.to_pickle( './data202310.pkl' )
