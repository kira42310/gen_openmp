import os
import re
import subprocess

from collections import deque

regex = re.compile( r'[\t]' )

def clean_comment_c( file_dir, output, timeout=20 ):
    command = f'gcc -fpreprocessed -dD -E -P {file_dir} > {output}'
    subprocess.run( command, timeout=timeout, shell=True)

def ast_dump_json( file_location, output, timeout=20 ):
    command = f'clang-16 -Xclang -ast-dump=json -fsyntax-only -fopenmp -ferror-limit=1000 {file_location} > {output}'
    subprocess.run( command, timeout=timeout, shell=True)

def ast_traverse_stack( data, loc = 1 ):

    stack = deque( ( item for item in reversed(data) if item != {} ) )

    del data

    return_data = []
    cur_loc = loc

    while len( stack ) > 0:

        tmp = stack.pop()

        if( 'range' in tmp.keys() and 'includedFrom' in tmp['range']['begin'].keys() ):
            continue

        if( 'range' in tmp.keys() ):
            if( 'line' in tmp['range']['begin'].keys() ):
                cur_loc = tmp['range']['begin']['line']

        if( 'kind' in tmp.keys() ):
            if( tmp['kind'] == 'ForStmt' or ( 'OMP' in tmp['kind'] and 'Directive' in tmp['kind'] ) ):

                if( 'inner' in tmp.keys() ):
                    if( tmp['inner'] != [{}] ):
                        inner_data = ast_traverse_stack( tmp['inner'], cur_loc )
                    else:
                        inner_data = []
                else:
                    inner_data = []

                if( 'line' in tmp['range']['end'].keys() ):
                    end_loc = tmp['range']['end']['line']
                else:
                    end_loc = cur_loc

                return_data.append({
                    'kind': tmp['kind'],
                    'start_line': cur_loc,
                    'end_line': end_loc,
                    'inner': inner_data
                })
                continue

        if( 'inner' in tmp.keys() ):
            for item in reversed(tmp['inner']):
                if( item != {} ):
                    stack.append( item )

    return return_data

def label_data( data, raw, file_path, file_nocom, d = 1 ):
    tmp_list = []
    if( 'OMP' in data['kind'] and 'Directive' in data['kind'] and 
      ( 'For' in data['kind'] or 'Simd' in data['kind'] or 'Loop' in data['kind'] or 'Distributed' in data['kind']  ) ):
        for inner_data in data['inner']:
            if( 'ForStmt' == inner_data['kind'] ):
                tmp_list.append({
                    'kind': data['kind'],
                    'omp_start': int( data['start_line'] ),
                    'omp_end': int( data['end_line'] ),
                    'for_start': int( inner_data['start_line'] ),
                    'for_end': int( inner_data['end_line'] ),
                    'parallel': 1,
                    'n_level': d,
                    'for_raw': regex.sub( '', ' '.join( raw[ int( inner_data['start_line'] ) - 1: int( inner_data['end_line'] ) ] ) ),
                    'omp_raw': regex.sub( '', ' '.join( raw[ int( data['start_line'] ) - 1: int( data['end_line'] ) ] ) ),
                    'file_path': file_path,
                    'file_nocom': file_nocom
                })
            for inner in inner_data['inner']:
                tmp_list.extend( label_data( inner, raw, file_path, file_nocom, d + 1) )

    elif( 'ForStmt' == data['kind'] ):
        tmp_list.append({
            'kind': data['kind'],
            'omp_start': 0,
            'omp_end': 0,
            'for_start': int( data['start_line'] ),
            'for_end': int( data['end_line'] ),
            'parallel': 0,
            'n_level': d,
            'for_raw': regex.sub( '', ' '.join( raw[ int( data['start_line'] ) - 1: int( data['end_line'] ) ] ) ),
            'omp_raw': None,
            'file_path': file_path,
            'file_nocom': file_nocom
        })
        for inner in data['inner']:
            tmp_list.extend(label_data( inner, raw, file_path, file_nocom, d + 1 ) )

    else:
        for inner in data['inner']:
            if d == 1:
                tmp_list.extend( label_data( inner, raw, file_path, file_nocom ) )
            else:
                tmp_list.extend( label_data( inner, raw, file_path, file_nocom, d ) )

    return tmp_list

def list_files( directory ):
    file_list = []
    for root, dirs, files in os.walk( directory ):
        files = [ f for f in files if not f[0] == '.' ]
        for name in files:
            path = os.path.join( root, name )
            if( os.path.isfile( path ) ):
                file_list.append( path )
    return file_list

def drop_dup( items ):
    new_items = []
    new_items.append( items[0] )
    for i in range( 1, len(items) ):
        if( items[i] not in new_items ):
            new_items.append(items[i])

    for i, item in enumerate( new_items ):
        # print( item )
        # print( '******************')
        if( len( item['inner'] ) > 0 ):
            new_items[i]['inner'] = drop_dup( item['inner'] )

    return new_items