import pandas as pd

from sctokenizer import CppTokenizer

directives = ['pragma', 'omp', 'parallel', 'for', 'sections', 'team' ]
# clauses = ['private', 'firstprivate', 'lastprivate', 'shared', 'default', 'reduction', 'schedule', 'ordered', 'num_threads', 'collapse' ]
clauses = ['private', 'firstprivate', 'lastprivate', 'shared', 'default', 'reduction', 'schedule', 'num_threads', 'collapse' ]
schedule = ['static', 'dynamic', 'guided', 'auto', 'runtime']


def tokens_to_list( tokens ):
    tmp = []
    for token in tokens:
        tmp.append( token.token_value )
    return tmp

tokenizer = CppTokenizer()
# gen_npb = []
# with open( 'output_gen_npb.txt', 'r' ) as f:
#     gen_npb = f.readlines()

# df = pd.read_pickle( '../data/training_data_filter_token3_ddup.pkl' )
# print( len(df) )
train_data_len = 33883
schedule_len = 8002
# omp_list = df['target'].tolist()
# print( df )

# count = {}
# for omp in omp_list:
#     tokens = tokenizer.tokenize( omp )
#     for token in tokens_to_list(tokens):
#         if( token in count ):
#             count[token] = count[token] + 1
#         else:
#             count[token] = 1

# print( count )

# df1 = pd.DataFrame( gen_npb )
# df2 = pd.DataFrame( count.values(), index=count.keys() )
# print( df2 )
# df1.to_pickle( 'gen_npb.pkl' )
# df2.to_pickle( 'ori_npb_count.pkl' )

df1 = pd.read_pickle( 'ori_npb_count.pkl' )
df2 = pd.read_pickle( 'gen_npb_count.pkl' )

# print( df1 )
# print( df2 )

filters = directives + clauses + schedule

# print( df1.loc[df1.index.intersection(directives+clauses).tolist()].sort_values(by=[0],ascending=False) )
train_ratio = df1.loc[df1.index.intersection( filters ).tolist()].sort_values(by=[0],ascending=False)
# print( df2.loc[df2.index.intersection(directives+clauses).tolist()].sort_values(by=[0],ascending=False) )
gen_ratio = df2.loc[df2.index.intersection( filters ).tolist()].sort_values(by=[0],ascending=False)

print( train_ratio )
print( gen_ratio )

train_ratio_clause = train_ratio.loc[ clauses ].sort_values( by=[0], ascending=False)
train_ratio_scheule = train_ratio.loc[ schedule ].sort_values( by=[0], ascending=False)
print( train_ratio_clause )
print( train_ratio_scheule )
print( train_ratio_scheule.values.sum() )

for i, d in train_ratio_clause.iterrows():
    print( i, round( d[0]/train_data_len, 4 ) )

for i, d in train_ratio_scheule.iterrows():
    print( i, round( d[0]/train_ratio_scheule.values.sum(), 4 ))