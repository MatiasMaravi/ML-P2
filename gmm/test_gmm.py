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
clases = pd.read_csv("class.txt")
clases = np.array(clases)
l = []
for x in clases:
	l.append(x[1])
l = set(l)
print(l)
data = np.array(dataset)




def test_gmm(x_size, K):

	pca = PCA(n_components=x_size)
	data_redimensionada = pca.fit_transform(data) 

	gmm_model = GMM(K)
	gmm_model.ajustar(data_redimensionada)
	clusters = gmm_model.clusters(data_redimensionada)
	return clusters

def show_silhoutte():
	k_values = [6,7,8]
	colores = ['red', 'green', 'blue', 'orange', 'lavender', 'yellow', 'pink', 'brown', 'gray', 'teal']
	for k in k_values:
		clusters = test_gmm(50,k)
		silhouette_avg = silhouette_score(dataset, clusters)
		sample_silhouette_values = silhouette_samples(dataset, clusters)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

		y_lower = 10
		for i in range(k):
			ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
			ith_cluster_silhouette_values.sort()
			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i
			color = colores[i % len(colores)]
			ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
			y_lower = y_upper + 10
			ax1.set_title("Silhouette plot for k={}".format(k))
			ax1.set_xlabel("Silhouette coefficient values")
			ax1.set_ylabel("Cluster label")
			ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
			ax1.set_yticks([])
			ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

			for cluster_id in range(k): 
				cluster_points = dataset[clusters == cluster_id]  
				ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colores[cluster_id], label=f'Cluster {cluster_id+1}')


			ax2.set_xlabel('Dimensión 1')
			ax2.set_ylabel('Dimensión 2')
			ax2.set_title('Visualización de Clústeres')
			ax2.legend()

			plt.show()
	

show_silhoutte()