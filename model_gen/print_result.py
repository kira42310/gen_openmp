import pandas as pd

ori_df = pd.read_pickle( '../data/eval_data_filter_token2_ddup.pkl' )
res_df = pd.read_pickle( 'noe_results.pkl' )

ori_list = ori_df['target'].tolist()
res_list = res_df.iloc[12]['generated_texts']

print( len( ori_list), len( res_list))

# for i in range( len( ori_list )):
#     print( '*************** ' + str(i) + ' ****************' )
#     # print( ori_df.iloc[i]['source'] )
#     print( ori_list[i] )
#     print( res_list[i] )

# 4221, 4224, 4225, 
# 4215, 4184
###############################
# 4232, 4233
print( ori_df.iloc[4232]['source'] )
print( ori_list[4232] )
print( res_list[4232] )
print( '*'*10)
print( ori_df.iloc[4233]['source'] )
print( ori_list[4233] )
print( res_list[4233] )

# code completion work 
# code clone detection tool ( ccfinder )
# join + split( )
# res_list[4233].isin( ori_list )
