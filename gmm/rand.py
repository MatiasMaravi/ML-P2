from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import numpy as np
df_true = pd.read_csv('./class.txt', index_col=0, usecols=[0, 1], skiprows=0)
labels_true =df_true.values.flatten()
rand_index = adjusted_rand_score(labels_true,labels)
print(f'Índice de Rand ajustado: {rand_index}')