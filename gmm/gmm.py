import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, dataset, max_iteraciones=100, tol=1e-4):
        self.dataset = dataset
        self.max_iteraciones = max_iteraciones
        self.tol = tol

    def ajustar(self, X):
        self.X = X
        self.n_muestras, self.n_caracteristicas = X.shape
        
        # Inicialización de parámetros
        self.pesos = np.ones(self.dataset) / self.dataset
        self.medias = X[np.random.choice(self.n_muestras, self.dataset, replace=False)]
        self.covarianzas = np.array([np.cov(X.T) for _ in range(self.dataset)])
        
        # Algoritmo EM
        for _ in range(self.max_iteraciones):
            # Etapa E
            responsabilidades = self._calcular_responsabilidades()
            
            # Etapa M
            suma_total = np.sum(responsabilidades, axis=0)
            self.pesos = suma_total / self.n_muestras
            self.medias = np.dot(responsabilidades.T, X) / suma_total[:, np.newaxis]
            self.covarianzas = np.array([np.dot((responsabilidades[:, k] * (X - self.medias[k]).T), (X - self.medias[k])) / suma_total[k]
                                          for k in range(self.dataset)])
            
            # Verificar convergencia
            if np.linalg.norm(self._log_verosimilitud(self.X) - self._log_verosimilitud(self.X, prev_params=True)) < self.tol:
                break
    
    def predecir(self, X):
        responsabilidades = self._calcular_responsabilidades(X)
        return np.argmax(responsabilidades, axis=1)
    
    def _calcular_responsabilidades(self, X=None):
        X = X if X is not None else self.X
        responsabilidades = np.array([self.pesos[k] * multivariate_normal.pdf(X, mean=self.medias[k], cov=self.covarianzas[k])
                                    for k in range(self.dataset)]).T
        responsabilidades /= np.sum(responsabilidades, axis=1)[:, np.newaxis]
        return responsabilidades
    
    def _log_verosimilitud(self, X, prev_params=False):
        if prev_params:
            responsabilidades = self._calcular_responsabilidades(X)
        else:
            responsabilidades = self._calcular_responsabilidades()
        return np.sum(np.log(np.sum(responsabilidades, axis=1)))