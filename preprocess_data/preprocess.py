import preprocess_util

import os
import sys
import signal
import json
import uuid
import time
import pandas as pd

from collections import deque

cur_location = '/work/soratouch-p/repos_data'
output_location = cur_location + '/output/'

file_list_location = cur_location + '/omp_file_list_2.pkl'

file_per_task = 20 # 145234 >> 7263 tasks ( 0 - 3999 + 4000 - 7263 )
TIMEOUT = 15 # second
GIGA = 1024 * 1024 * 1024

log_file = cur_location + '/error/log.log'
err_file = cur_location + '/error/error.log'

def timeout_handler( num, stack ):
    raise Exception('Timeout')

### for clean comment 
if __name__ == '__main__':
    task_id = sys.argv[1]
    tmp_ast = f'{cur_location}/tmp_file_{task_id}.json'

    #source_df = pd.read_pickle( file_list_location ).reset_index( drop=True )
    source_df = pd.read_pickle( file_list_location )

    start = int( task_id ) * file_per_task
    end = ( int( task_id ) + 1 ) * file_per_task
    if( end < len( source_df ) ):
        df = source_df[ start: end ]
    else:
        df = source_df[ start: len( source_df ) ]

    # df = df[0:]

    signal.signal( signal.SIGALRM, timeout_handler )

    for i, item in df.iterrows():

        ############### Clean Comment #############
        file_path = item['path'][10:]
        tmp_path = file_path.split('/')
        tmp_name = tmp_path[-1].split('.')
        if( len( tmp_name ) == 2 ):
            tmp_path[-1] = tmp_name[0] + '_nocomment.' + tmp_name[1]
        else:
            tmp_path[-1] = '.'.join(tmp_name[:-1]) + '_nocomment.' + tmp_name[-1]
        input_file = cur_location + file_path
        clean_comment_file = cur_location + '/'.join( tmp_path )
        try:
            preprocess_util.clean_comment_c( input_file, clean_comment_file )
        except:
            with open( err_file, 'a' ) as err:
                err.write( f'CLEAN TASKID:{task_id} [{i}] {file_path}\n' )
            continue

        ################ AST DUMP ##############
        if( os.path.exists( tmp_ast ) ):
            os.remove( tmp_ast )
            
        time.sleep(1)
        signal.alarm( TIMEOUT )
        try:
            preprocess_util.ast_dump_json( clean_comment_file, tmp_ast )
        except:
            signal.alarm( 0 )
            with open( err_file, 'a' ) as err:
                err.write( f'ASTDUMP TASKID:{task_id} [{i}] {file_path}\n')
            if( os.path.exists( tmp_ast ) ):
                os.remove( tmp_ast )
            continue
        finally:
            signal.alarm( 0 )
            time.sleep(1)
        
        ################# TRAV AST ###############
        file_stats = os.stat( tmp_ast )
        if( ( file_stats.st_size / GIGA ) > 10 ):
            with open( err_file, 'a' ) as err:
                err.write( f'TRAVERSE TASKID:{task_id} [{i}] {file_path}\n')
            continue
        
        try:
            with open( tmp_ast, 'r' ) as j_f:
                json_data = json.load( j_f )

            ast_stack = deque( json_data['inner'] )
            del json_data
            if( os.path.exists( tmp_ast ) ):
                os.remove( tmp_ast )
            ast = preprocess_util.ast_traverse_stack( ast_stack )
        except:
            with open( err_file, 'a' ) as err:
                err.write( f'TRAVERSE TASKID:{task_id} [{i}] {file_path}\n')
            continue

        ################## WRITE FILE #############
        try:
            filename = str( uuid.uuid4() )
            if( os.path.exists( filename ) ):
                filename = str( uuid.uuid4() )

            tmp_data = { 'file_path': file_path, 'data': ast }
            pd.DataFrame( tmp_data ).to_pickle( output_location + filename + '.pkl' ) 
        except:
            with open( err_file, 'a' ) as err:
                err.write( f'WRITE TASKID:{task_id} [{i}] {file_path}\n')

        if( os.path.exists( tmp_ast ) ):
            os.remove( tmp_ast )

        with open( log_file, 'a' ) as l_f:
            l_f.write( f'TASKID:{task_id} [{i}] {file_path}\n')
