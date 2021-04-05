import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("kkbox_2.csv")
df.dropna(inplace = True)
# y = df['duration']
# df.drop(['duration'])
# X = df
# X = train_test_split(df, test_size = 0.0037, random_state = 0, stratify = )
N = 5000
df_sample = df.groupby('duration', group_keys=False).apply(lambda x: x.sample(int(np.rint(N*len(x)/len(df))))).sample(frac=1).reset_index(drop=True)
print(df_sample.head)
df_sample.drop(['Unnamed: 0'], axis=1, inplace = True)
print(df_sample.columns)
df_sample.to_csv("kkbox_sample.csv")