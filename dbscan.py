import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from sklearn.metrics import silhouette_score
def dist(a, b):  # Distancia euclidiana
    return np.sqrt(np.sum((a - b) ** 2))

def range_query(DB,dist,p1,radius):
    neighbors = []
    for p2 in range(len(DB)):
        if dist(p1,DB[p2]) <= radius:
            neighbors.append(p2)
    return neighbors

def dbscan_with_kdtree(DB, radius, minPts, kdtree):
    label = [None] * len(DB)
    cluster = -1
    for p in range(len(DB)):
        if label[p] is not None:
            continue
        
        # Encuentra los vecinos dentro del radio utilizando el KD-Tree
        neighbors_indices = kdtree.query_ball_point(DB[p], radius)
        
        if len(neighbors_indices) < minPts:
            label[p] = -1
            continue
        
        cluster += 1
        label[p] = cluster
        S = set(neighbors_indices)
        while S:
            q = S.pop()
            if label[q] == -1:
                label[q] = cluster
            if label[q] is not None:
                continue
            
            # Encuentra los vecinos dentro del radio utilizando el KD-Tree
            neighbors_indices = kdtree.query_ball_point(DB[q], radius)
            label[q] = cluster
            if len(neighbors_indices) >= minPts:
                S.update(neighbors_indices)
    return label

def porcentaje_ruido(label):
    count = np.count_nonzero(label == -1)
    return count/len(label) * 100

import pandas as pd

df_tissue = pd.read_csv('./dataset/dataset_tissue.txt',sep=',',index_col=0)
df_clases = pd.read_csv('./dataset/clase.txt',sep=',',index_col=0)
# Obtener los nombres de las clases del DataFrame
tissue_names = df_clases['x'].tolist()

def graficar_silueta(idx_eps,idx_samples,X,label,nombre_archivo):
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples

    # Calcula las puntuaciones de Silhouette para cada muestra
    silhouette_values = silhouette_samples(X, label)

    # Calcula la puntuación de Silhouette promedio para todos los datos
    silhouette_avg = silhouette_score(X, label)

    # Crea un gráfico de barras para mostrar las puntuaciones de Silhouette individuales
    fig, ax = plt.subplots(figsize=(10, 6))

    y_lower = 10
    for i in np.unique(label):
        cluster_silhouette_values = silhouette_values[label == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(i / len(np.unique(label)))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Etiqueta para cada cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Calcula el siguiente y_lower para el próximo cluster en el gráfico
        y_lower = y_upper + 10

    # Línea vertical para la puntuación de Silhouette promedio de todos los datos
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Etiqueta para la puntuación de Silhouette promedio
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel("Puntuación de Silhouette")
    plt.title(f"Puntuación de Silhouette para cada cluster (Promedio: {silhouette_avg:.2f})")
    # Añadir texto a la esquina derecha
    plt.text(0.95, 0.95, f'eps: {idx_eps}\nmin_samples: {idx_samples}',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,  # Para especificar coordenadas relativas al sistema de ejes
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))  # Opcional: un cuadro para resaltar el texto

    plt.savefig(f'images/{nombre_archivo}')
    plt.close(fig)


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

df_tissue_transposed = df_tissue.T

n_components =70
pca = PCA(n_components=n_components)
dt_tissue_pca = pca.fit_transform(df_tissue_transposed)
print(dt_tissue_pca.shape)
# max, idx_eps, idx_samples = 0, 0, 0
# contador = 0
# for i in range(3,6):
#     j = 5.02
#     while j < 100:
#         dbscan = DBSCAN(eps=j, min_samples=i,metric='euclidean',algorithm='kd_tree')
#         label = dbscan.fit_predict(dt_tissue_pca)
#         # Filtra los puntos de ruido (label == -1)
#         # Verifica si se generan al menos dos clusters
#         if(label[label == -1].size > 6 or len(np.unique(label)) < 8 or len(np.unique(label)) > 9):
#             j+=0.02
#             continue
#         idx_eps = j
#         idx_samples = i
#         nombre = f"archivo_{contador}"
#         graficar_silueta(idx_eps,idx_samples,dt_tissue_pca,label,nombre)
#         print("Silueta ",contador," creada exitosamente")
#         contador+=1
#         j+=0.02

# dbscan = DBSCAN(eps=77.0600000000182, min_samples=3,metric='euclidean',algorithm='kd_tree')
label = dbscan_with_kdtree(dt_tissue_pca, 77.0600000000182, 3, KDTree(dt_tissue_pca))
print("-----------------LABEL----------------")
print(label)
def hallar_indices(label):
    dict_idx = {}
    for i in range(len(label)):
        if(label[i] not in dict_idx):
            dict_idx[label[i]] = [i]
        else:
            dict_idx[label[i]].append(i)
    return dict_idx

lista = []
dict_real = {
    "kidney":0,
    "hippocampus":1,
    "cerebellum":2,
    "colon":3,
    "liver":4,
    "endometrium":5,
    "cerebellum":6,
    "placenta":7,
}
dict_idx = hallar_indices(label)
for i in dict_idx[0]:
    lista.append(tissue_names[i])

print(lista)

new_label = []
for i in label:
    if(i == 6):
        new_label.append(2)
    else:
        new_label.append(i)



from sklearn.metrics import adjusted_rand_score
diccionario = {
    "Ruido":-1,
    "kidney":0,
    "hippocampus":1,
    "cerebellum":2,
    "colon":3,
    "liver":4,
    "endometrium":5,
    "placenta":7
}
# Suponiendo que 'labels_true' son las etiquetas reales y 'labels_pred' son las etiquetas predichas
labels_true = []
for i in range(len(new_label)):
    labels_true.append(diccionario[tissue_names[i]])

print("-----------------LABELS TRUE----------------")
print(labels_true)
print("-----------------NEW LABEL----------------")
print(new_label)
rand_index = adjusted_rand_score(labels_true, new_label)

print("Indice rand: ",rand_index)

#Cohesion
from sklearn.metrics import pairwise_distances
cohesion_values = []
for i in np.unique(new_label):
    cluster_points = dt_tissue_pca[new_label == i]
    # Calcular la matriz de distancias entre puntos dentro del mismo clúster
    distances_within_cluster = pairwise_distances(cluster_points, metric='euclidean')
    # Calcular la cohesión promedio para el clúster actual
    mean_cohesion = np.mean(distances_within_cluster)
    cohesion_values.append(mean_cohesion)

print("Cohesion: ",np.mean(cohesion_values))

#Separacion
# Calcular la separación
separation_values = []
for label1 in np.unique(new_label):
    for label2 in np.unique(new_label):
        if label1 != label2:
            cluster1_points = dt_tissue_pca[new_label == label1]
            cluster2_points = dt_tissue_pca[new_label == label2]
            # Calcular la matriz de distancias entre puntos de diferentes clústeres
            distances_between_clusters = pairwise_distances(cluster1_points, cluster2_points, metric='euclidean')
            # Calcular la separación promedio entre los clústeres actuales
            mean_separation = np.mean(distances_between_clusters)
            separation_values.append(mean_separation)

print("Separacion: ",np.mean(separation_values))

rango_total = np.ptp(dt_tissue_pca)
print("Rango total:", rango_total)


#Mutual information
from sklearn.metrics.cluster import normalized_mutual_info_score as mi

mutual_information = mi(labels_true, new_label)
print("Mutual information: ",mutual_information)



#Matriz de similitud
# from sklearn.metrics import pairwise_distances
# # Ordenar los datos según su etiqueta de cluster
# sorted_data = dt_tissue_pca[np.argsort(new_label)]
# # Calcular la matriz de similitud
# similarity_matrix = pairwise_distances(sorted_data, metric='euclidean')

# # Mostrar la matriz de similitud
# plt.imshow(similarity_matrix, cmap='jet')
# plt.colorbar()
# plt.show()
#DENDOGRAMA

# import numpy as np
# from scipy.spatial.distance import pdist
# from scipy.cluster.hierarchy import linkage, dendrogram
# import matplotlib.pyplot as plt

# # Ejemplo de datos (matriz de distancias)
# data = np.array(dt_tissue_pca)

# # Calcular la matriz de distancias a partir de los datos
# distances = pdist(data)

# # Realizar el clustering jerárquico
# linkage_matrix = linkage(distances, method='complete')

# # Crear el dendrograma
# dendrogram(linkage_matrix,
#            labels=label,
#            truncate_mode='level',
#            p=8,
#            leaf_rotation=90,
#            leaf_font_size=6)
# plt.xlabel('Observaciones')

# plt.ylabel('Distancia')

# # Mostrar el dendrograma
# plt.show()




# count = 0
# for i in range(len(label)):
#     if(label[i] == diccionario[tissue_names[i]]):
#         count+=1

# print("porcentaje de acierto: ",count/len(label) * 100)
# Utiliza la paleta de colores prediseñada "deep" de seaborn
# colors = sns.color_palette("muted", len(np.unique(label)))

# # Crea un gráfico de dispersión con colores basados en los clusters
# for cluster_label in np.unique(label):
#     if cluster_label == -1:
#         # El cluster -1 es el ruido, color negro
#         color = [0,0,0]
#     else:
#         color = colors[cluster_label]  # selecciona el color según la etiqueta del cluster
#     cluster_points = dt_tissue_pca[label == cluster_label]  # puntos en el cluster actual
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster_label}')
# plt.xlabel('Dimensión Principal 1')
# plt.ylabel('Dimensión Principal 2')
# plt.legend()
# plt.show()


# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.cluster import AgglomerativeClustering
