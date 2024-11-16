import pandas as pd

pickle_location = '../data/npb.pkl'

pd = pd.read_pickle( pickle_location )

print( pd )