import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle( 'token3_results.pkl' )

print( df )

scores = pd.DataFrame( df[ 'scores' ].tolist() )

print( scores )

scores.plot( kind='bar' )

ax = plt.gca()
ax.set_xticklabels( ( '4', '8', '12', '16', '20', '24', '28', '32', '36', '40', '44', '48', '52' ) )

plt.ylim( (0.6, 1))
plt.legend( loc=4 )
plt.xlabel( 'Step(s)' )
plt.ylabel( '%' )

plt.savefig( 'res.pdf', format='pdf')