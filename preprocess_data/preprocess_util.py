import os
import re
import subprocess
import pandas as pd

from collections import deque

regex = re.compile( r'[\t]' )

def pprint_ast( datas, n_level = 1 ):
    for data in datas:
        print( f"{'|'*(n_level-1)}-{data['kind']}, {data['start_line']}, {data['end_line']}" )
        pprint_ast( data['inner'], n_level=n_level+1 )

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

def drop_dup_pd( items ):
    return items.sort_values( [ 'for_start', 'for_end', 'parallel' ], ascending=[ True, True, False ] ) \
            .drop_duplicates( subset=['for_start', 'for_end'] ) \
            .reset_index( drop=True )

def add_stars_reponame_to_pd( data ):
    stars = []
    repo_name = []
    tmp = data['file_path']
    for path in tmp:
        p = path.split( '/' )
        stars.append( p[2] )
        repo_name.append( '/'.join( [ p[3],p[4] ] ) )
    data['stars'] = stars
    data['repo_name'] = repo_name
    return data

def add_tokens_to_pd( data ):
    from sctokenizer import CppTokenizer
    tokenizer = CppTokenizer()
    tokens_for_list = []
    count_for_list = []
    tokens_omp_list = []
    count_omp_list = []
    for i, d in data.iterrows():
        if d['parallel'] == 1:
            raw_for = d['for_raw']
            raw_omp = d['omp_raw'] 
            tokens_for = tokenizer.tokenize( raw_for )
            tokens_omp = tokenizer.tokenize( raw_omp )
            tmp_for = [ token.token_value for token in tokens_for ]
            tmp_omp = [ token.token_value for token in tokens_omp ]
            tokens_for_list.append( tmp_for )
            tokens_omp_list.append( tmp_omp )
            count_for_list.append( len( tmp_for ) )
            count_omp_list.append( len( tmp_omp ) )
        else:
            raw_for = d['for_raw']
            tokens_for = tokenizer.tokenize( raw_for )
            tmp_for = [ token.token_value for token in tokens_for ]
            tokens_for_list.append( tmp_for )
            tokens_omp_list.append( None )
            count_for_list.append( len( tmp_for ) )
            count_omp_list.append( 0 )
    data['tokens_for'] = tokens_for_list
    data['tokens_count_for'] = count_for_list
    data['tokens_omp'] = tokens_omp_list
    data['tokens_count_omp'] = count_omp_list
    return data

def get_frac( stars ):
    s = str( stars )
    if s == '4':
        return 0.01
    elif s == '3':
        return 0.05
    elif s == '2':
        return 0.1
    return 0.1


def create_training_data_generative( items, frac: float = 1, seed: int = 4, filter: list = None ):
    tmp_df = items[items['parallel'] == 1]
    tmp_df = tmp_df[ tmp_df['tokens_count_for'] > 20 ]
    tmp_df = tmp_df if filter is None else tmp_df[tmp_df['kind'].isin(filter)]

    ### seperate by repo name ###
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    for name in tmp_df['repo_name'].unique():
        repo_df = tmp_df[ tmp_df['repo_name'] == name ]
        stars = str( repo_df.iloc[0]['stars'] )
        if( stars == '5' ):
            repo_df = repo_df.drop_duplicates( subset=['for_raw', 'omp_raw'] )
            train_df = pd.concat( [ train_df, repo_df ] )
        else:
            frac_v = 1 - get_frac( stars )
            repo_df = repo_df.drop_duplicates( subset=['for_raw'] )
            tmp_train_df = repo_df.sample( frac=frac_v, random_state=seed )
            tmp_eval_df = repo_df.drop( tmp_train_df.index )
            train_df = pd.concat( [ train_df, tmp_train_df ] )
            eval_df = pd.concat( [ eval_df, tmp_eval_df ] )

            # for_dup = repo_df.duplicated( subset=['for_raw'] ).value_counts().sort_index()
            # for_dup_count = ( ( 0 if for_dup.index.tolist()[0] == False else for_dup[0] ) if len( for_dup ) == 1 else for_dup[1] )
            # all_dup = repo_df.duplicated( subset=['for_raw', 'omp_raw'] ).value_counts().sort_index()
            # all_dup_count = ( ( 0 if all_dup.index.tolist()[0] == False else for_dup[0] ) if len( all_dup ) == 1 else all_dup[1] )
            # # print( for_dup_count, for_dup_count*1.111, all_dup_count )
            # if( for_dup_count == 0 and all_dup_count == 0 ):
            #     train_df = pd.concat( [ train_df, repo_df ] )
            # elif( all_dup_count * 1.111 ) >= for_dup_count:
            #     repo_df = repo_df.drop_duplicates( subset=['for_raw'] )
            #     train_df = pd.concat( [ train_df, repo_df ] )
            # else:
            #     frac = get_frac( stars )
            #     repo_df = repo_df.drop_duplicates( subset=['for_raw', 'omp_raw'] )
            #     tmp_eval_df = repo_df[repo_df.duplicated( subset=['for_raw'] ) ]
            #     tmp_train_df = repo_df.drop( tmp_eval_df.index )
            #     tmp_eval_df = tmp_eval_df.sample( frac=frac )
            #     # tmp_leftover = dup_df.drop( tmp_eval_df.index )
            #     # tmp_train_df = pd.concat( [ tmp_train_df, tmp_leftover ] )
            #     # train_df = pd.concat( [ train_df, tmp_train_df, tmp_leftover ] )
            #     train_df = pd.concat( [ train_df, tmp_train_df ] )
            #     eval_df = pd.concat( [ eval_df, tmp_eval_df ] )

        # new_df = pd.concat( [new_df,repo_df] )

    # train_df = train_df[['for_raw','omp_raw']].rename( columns={ 'for_raw': 'source', 'omp_raw': 'target' } ).reset_index( drop=True )
    # eval_df = eval_df[['for_raw','omp_raw']].rename( columns={ 'for_raw': 'source', 'omp_raw': 'target' } ).reset_index( drop=True )
    train_df = train_df[['for_raw','omp_raw','tokens_count_for','stars', 'repo_name']].rename( columns={ 'for_raw': 'source', 'omp_raw': 'target' } ).reset_index( drop=True )
    eval_df = eval_df[['for_raw','omp_raw','tokens_count_for', 'stars', 'repo_name']].rename( columns={ 'for_raw': 'source', 'omp_raw': 'target' } ).reset_index( drop=True )
    return {
        'train_data': train_df,
        'eval_data': eval_df
    }

    # tmp_df = tmp_df[['for_raw','omp_raw']]
    # tmp_df = tmp_df[ ~tmp_df['omp_raw'].isnull() ]
    # tmp_df = tmp_df.rename( columns={ 'for_raw': 'source', 'omp_raw': 'target' } )
    # tmp_df = tmp_df.drop_duplicates()
    # tr_df = tmp_df.sample( frac=frac, random_state=seed )
    # ev_df = tmp_df.drop( tr_df.index ).reset_index( drop=True )
    # tr_df = tr_df.reset_index( drop=True )
    # return {
    #     'train_data': tr_df,
    #     'eval_data': ev_df
    # }

def create_training_data_classification( items ):
    tmp_df = items[['for_raw','parallel']]
    tmp_df = tmp_df.rename( columns={ 'for_raw': 'source', 'parallel': 'target' } )
    return tmp_df

def fix_space_nline( data ):
    return ' '.join( ' '.join( data.split('\n') ).split() )