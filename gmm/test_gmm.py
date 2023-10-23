import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from sklearn.decomposition import PCA
from gmm import GMM
dataset = pd.read_csv("dataset_tissue.txt")
dataset.drop(["Unnamed: 0"], axis=1, inplace=True)

dataset = dataset.T

pca = PCA(n_components=70)
dataset = pca.fit_transform(dataset) 
clases = pd.read_csv("class.txt")
clases = np.array(clases)
l = []
for x in clases:
	l.append(x[1])
l = set(l)



def test_gmm(K):
	gmm_model = GMM(K)
	gmm_model.ajustar(dataset)
	clusters = gmm_model.clusters(dataset)
	return clusters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

# Train the GMM model

data_redimensionada = dataset;

for k in [6,7,8]:
	gmm = GMM(k)
	gmm.ajustar(data_redimensionada)

	# Calculate the silhouette scores
	labels = gmm.predecir(data_redimensionada)
	silhouette_vals = silhouette_samples(data_redimensionada, labels)

	# Calculate the average silhouette score
	silhouette_avg = silhouette_score(data_redimensionada, labels)
	print("Silhouette score:", silhouette_avg)
	# Plot the silhouette scores
	y_lower, y_upper = 0, 0
	fig, ax = plt.subplots()
	for i in range(gmm.K):
	    ith_cluster_silhouette_vals = silhouette_vals[labels == i]
	    ith_cluster_silhouette_vals.sort()
	    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
	    y_upper += size_cluster_i
	    ax.barh(range(y_lower, y_upper), ith_cluster_silhouette_vals, height=1.0)
	    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
	    y_lower += size_cluster_i

	ax.axvline(x=silhouette_avg, color="red", linestyle="--")
	ax.set_yticks([])
	ax.set_xlim([-0.1, 1])
	ax.set_xlabel("Silhouette coefficient values")
	ax.set_ylabel("Cluster labels")
	plt.show()

