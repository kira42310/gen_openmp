import pandas as pd


# bms = ['bt','cg','ep','ft','is','lu','mg','sp']
# for bm in bms: 
#     ser = pd.read_pickle( '../../npb-prep/ast/'+bm+'.cpp.pkl' )
#     omp = pd.read_pickle( '../../npb-prep/ast/'+bm+'.cppomp.pkl' )

#     print( f'ser:{len(ser)}, omp:{len(omp)}')

omp = pd.read_pickle( '../../npb-prep/ast/lu.cppomp.pkl' )
# outer_par =  omp[(omp['n_level'] == 1) & (omp['parallel'] == 1) ]

# print( omp[omp['parallel'] == 1 ] )
outer_par = omp[omp['parallel'] == 1 ] 

print( outer_par[['for_start','for_end', 'for_raw']] )

ser = pd.read_pickle( '../../npb-prep/ast/lu.cpp.pkl' )
# outer_ser = ser[ ser['n_level'] == 1 ]

print( ser[['for_start','for_end','for_raw']] )
# print( ser.iloc[[5, 12, 66, 69, 143, 144, 147, 153 ]][['for_start','for_end','for_raw']] )

# i = 15
# print( omp.iloc[i]['for_raw'] )
# print( ser.iloc[17]['for_raw'])
# CG
# 1, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19
# FT
# 3, 4, 5, 8, 9, 13, 15, 23
# IS
# 3, 5, 7, 10, 15, 18
# Lu
# 2, 5, 9, 12, 16, 20, 24, 35, 46, 64, 66, 69, 90, 94, 105, 119, 131, 134, 137, 140, 147, 153
# MG
# 8, 10, 12, 14, 21, 28, 37, 40, 44, 48, 61
insert_data = {
    'kind': 'ForStmt',
    'omp_start': 0,
    'omp_end': 0,
    'for_start': 2171,
    'for_end': 2182,
    'parallel': 0,
    'n_level': 1,
    'for_raw': '''for(j=0; j<ISIZ2; j++){
for(i=0; i<ISIZ1; i++){
for(n=0; n<5; n++){
for(m=0; m<5; m++){
a[j][i][n][m]=0.0;
b[j][i][n][m]=0.0;
c[j][i][n][m]=0.0;
d[j][i][n][m]=0.0;
}
}
}
}
    ''',
    'omp_raw': None,
    'file_path': 'LU/lu.cpp',
    'file_nocom': '-'
}
filter_ser = ser[ ser['for_raw'].isin(outer_par['for_raw'])]
# BT
# filter_ser = pd.concat( [filter_ser, outer_ser.loc[[155,163,171]]])
# SP
# filter_ser = pd.concat( [filter_ser, ser.loc[[167,194,221]]])
# cg_filter = [ 1, 3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 17, 19 ]
# ft_filter = [ 3, 4, 5, 8, 9, 13, 15, 23 ]
# is_filter = [ 3, 5, 7, 10, 15, 18 ]
# lu_filter = [ 2, 5, 9, 12, 16, 20, 24, 35, 46, 64, 66, 69, 90, 94, 105, 119, 131, 134, 137, 140, 147, 153 ]
# # mg_filter = [ 8, 10, 12, 14, 21, 28, 37, 40, 44, 48, 61 ]
# filter_ser = ser.loc[lu_filter]
# filter_ser = pd.concat([ filter_ser, pd.DataFrame([insert_data])])
print( filter_ser )

# output = '../../npb-prep/filter/lu.pkl' 

# filter_ser.to_pickle( output )

# print( pd.read_pickle( output ) )