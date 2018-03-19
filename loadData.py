import pandas as pd

df = pd.read_pickle('ppn1-downsampled.pkl')
print df

frontal = df['FCz']

for i in frontal.index:
    print i


#print(df[df['condition']=='65304'])

