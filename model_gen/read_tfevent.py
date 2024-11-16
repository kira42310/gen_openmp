# from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd

# tmp = []
# for summary in summary_iterator("saved_models/gen_filter_token3_ddup/events.out.tfevents.1712641174.RTX4090"):
    # tmp.append( summary )
    # tmp.append( { 'steps': int(summary.step), 'wall_time': str(summary.wall_time) , 'value': str(summary.summary.value) } )

# print( )
# print( len(tmp) )
# print( '*' * 10 )
# print( type(str(tmp[4127].summary.value))  )

# chk = [ '1058', '2116', '3174', '4232', '5290', '6348', '7406', '8464', '9522', '10580', '11638', '12696' ]
chk = [ 1, 1060, 2120, 3180, 4230, 5290, 6350, 7410, 8470, 9520, 10580, 11640, 12700, 13728 ]

# pd.DataFrame(tmp).to_pickle( 'rawlog_gen_filter_token3_ddup_tfevents.pkl' )
# df = pd.read_pickle( 'rawlog_gen_filter_token3_ddup_tfevents.pkl' )

# lasttime = 0

# for i, d in df.iterrows():
#     # print( d['steps'] )
#     if d['steps'] in chk:
        
#         print( d['steps'], (float(d['wall_time']) - lasttime)/60.0 ) 
#         lasttime = float(d['wall_time'])

# print( '*' * 10 )
# print( ( float( df.iloc[-1]['wall_time'] ) - float( df.iloc[0]['wall_time'] ) ) / 3600.0 )

df = pd.read_pickle( 'token3_results.pkl' )

for i, d in df.iterrows():
    print( len(d['generated_texts']))
    print( d['scores'] )

print( df )
# print( df.iloc[0]['generated_texts'] )