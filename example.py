from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
# Generar datos de ejemplo con dos lunas
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters1 = dbscan.fit_predict(X)
print(clusters1)

def dist(a, b):  # Distancia euclidiana
    return np.sqrt(np.sum((a - b) ** 2))

def range_query(DB,dist,p1,radius):
    neighbors = []
    for p2 in range(len(DB)):
        if dist(p1,DB[p2]) <= radius:
            neighbors.append(p2)
    return neighbors

def dbscan(DB, radius, minPts, dist):
    label = [None]*len(DB) # tamanio 189
    cluster = -1
    for p in range(len(DB)):
        if label[p] is not None:
            continue
        neighbors = range_query(DB,dist,DB[p],radius)
        if len(neighbors) < minPts:
            label[p] = -1
            continue
        cluster += 1
        label[p] = cluster
        S = set(neighbors) - {p}
        while S:
            q = S.pop()
            if label[q] == -1:
                label[q] = cluster
            if label[q] is not None:
                continue
            neighbors = range_query(DB,dist,DB[q],radius)
            label[q] = cluster
            if len(neighbors) >= minPts:
                S.update(neighbors.copy())
    return label

clusters = dbscan(X, 0.3, 5, dist)
print(np.array(clusters) == clusters1)