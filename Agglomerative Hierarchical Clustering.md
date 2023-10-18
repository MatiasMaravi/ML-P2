## Agglomerative Hierarchical Clustering

El clustering jerárquico aglomerativo es una técnica popular utilizada en el análisis de datos y el aprendizaje automático para agrupar puntos de datos similares en función de sus similitudes mutuas. Es un enfoque de abajo hacia arriba, o "aglomerativo", para la agrupación. El proceso comienza con cada punto de datos como su propio grupo y luego fusiona grupos de manera iterativa en grupos más grandes hasta que se cumple un criterio de parada, lo que suele resultar en un solo grupo que contiene todos los puntos de datos.

### Ejemplos

```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering().fit(X)

clustering.labels_ // array([1, 1, 1, 0, 0, 0])
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))
print(data) // [(4, 21), (5, 19), (10, 24), (4, 17), (3, 16), (11, 25), (14, 24), (6, 22), (10, 21), (12, 21)]

linkage_data = linkage(data, method='ward', metric='euclidean')

dendrogram(linkage_data)
plt.show()

labels = hierarchical_cluster.fit_predict(data)
print(labels) //[0 0 1 0 0 1 1 0 1 1]

```
