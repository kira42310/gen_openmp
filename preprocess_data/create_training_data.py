import sys

import preprocess_util

import pandas as pd

omp_directive_filter = [
    'OMPParallelForDirective',
    'OMPForDirective',
    # 'OMPSimdDirective',
    'OMPTargetTeamsDistributeParallelForDirective',
    'OMPParallelForSimdDirective',
    # 'OMPTargetTeamsDistributeSimdDirective',
    'OMPTargetTeamsDistributeParallelForSimdDirective',
    'OMPForSimdDirective',
    # 'OMPTargetSimdDirective',
    'OMPTargetParallelForSimdDirective',
    'OMPTargetParallelForDirective',
    'OMPDistributeParallelForSimdDirective',
    # 'OMPTeamsDistributeSimdDirective',
    # 'OMPTaskLoopDirective',
    'OMPTeamsDistributeParallelForDirective',
    'OMPTeamsDistributeParallelForSimdDirective',
    'OMPDistributeParallelForDirective',
    # 'OMPDistributeSimdDirective',
    # 'OMPTaskLoopSimdDirective',
    # 'OMPMasterTaskLoopSimdDirective',
    # 'OMPParallelMasterTaskLoopSimdDirective',
    # 'OMPMasterTaskLoopDirective',
    # 'OMPParallelMasterTaskLoopDirective',
    # 'OMPGenericLoopDirective',
    # 'OMPParallelGenericLoopDirective',
    # 'OMPTargetTeamsGenericLoopDirective',
    # 'OMPTeamsGenericLoopDirective',
    # 'OMPTargetParallelGenericLoopDirective',
    # 'OMPMaskedTaskLoopSimdDirective',
    # 'OMPParallelMaskedTaskLoopSimdDirective',
    # 'OMPParallelMaskedTaskLoopDirective',
    # 'OMPMaskedTaskLoopDirective'
]

load_file = '../data/data202404_stack_v6.pkl'
output_location = '../data'

if len(sys.argv) < 2:
    print( 'select options!!!! c or g' ) 
    exit()

train_type = sys.argv[1]
# frac = float( sys.argv[2] )
# seed = int( sys.argv[3] )

df = pd.read_pickle( load_file )

if train_type == 'g':
    data = preprocess_util.create_training_data_generative( df, frac=1, seed=4, filter=omp_directive_filter )
    # data.to_pickle( output_location + '/training_data_new_filter.pkl' )
    print( data['train_data'] )
    print( data['eval_data'] )
    # data['train_data'].sample( frac=1, random_state=4 ).to_pickle( output_location + '/training_data_filter_token2.pkl' )
    data['train_data'] = data['train_data'].sample( frac=1, random_state=4 ).reset_index( drop=True )
    data['train_data'].to_pickle( output_location + '/training_data_filter_token3.pkl' )
    data['eval_data'].to_pickle( output_location + '/eval_data_filter_token3.pkl' )
elif train_type == 'c':
    train_df = preprocess_util.create_training_data_classification( df )
    train_df.to_pickle( output_location + '/training_data.pkl' )
else:
    print( 'select options!!!! c or g' )

