import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from gmm import GMM
dataset = pd.read_csv("dataset_tissue.txt")
dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
# Train the GMM model
dataset = dataset.T

pca = PCA(n_components=70)
data_redimensionada = pca.fit_transform(dataset) 
gmm = GMM(K=7)
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


from sklearn.metrics.cluster import adjusted_rand_score

df_true = pd.read_csv('./class.txt', index_col=0, usecols=[0, 1], skiprows=0)
labels_true =df_true.values.flatten()
rand_index = adjusted_rand_score(labels_true,labels)
print(f'Índice de Rand ajustado: {rand_index}')


from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np

# Sort the data according to their predicted cluster labels
sorted_data = data_redimensionada[np.argsort(labels)]

# Calculate the pairwise distances
similarity_matrix = pairwise_distances(sorted_data, metric='euclidean')

# Plot the similarity matrix
plt.imshow(similarity_matrix, cmap='jet')
plt.colorbar()
plt.show()