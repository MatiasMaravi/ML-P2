import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from gmm import GMM
dataset = pd.read_csv("dataset_tissue.txt")
dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
dataset = dataset.T


data = np.array(dataset)

media = np.mean(data, axis=0)
desviacion_estandar = np.std(data, axis=0)
data_normalizada = (data - media) / desviacion_estandar

n_components = 100

pca = PCA(n_components=n_components)
data_redimensionada = pca.fit_transform(data_normalizada) 

gmm_model = GMM(data_redimensionada)