import numpy as np
from scipy.stats import multivariate_normal
np.random.seed(135)



class GMM:
    def __init__(self, K, max_iteraciones=60, tol=1e-4):
        self.K = K
        self.max_iteraciones = max_iteraciones
        self.tol = tol

    def calcular_log_verosimilitud(self, dataset, pesos, medias, covarianzas):
        verosimilitud = 0
        for d in dataset:
            tot = [self.pesos[i] * multivariate_normal.pdf(d, mean=self.medias[i], cov=self.covarianzas[i],allow_singular = True) for i in range(self.K)]
            verosimilitud += np.log(sum(tot))
        return verosimilitud
        
    def inicializar_parametros(self, dataset):
        
        muestras, caracteristicas = dataset.shape
        
      
        self.pesos = np.ones(self.K) / self.K  
        centroides_indices = np.random.choice(muestras, self.K, replace=False)
        centroides = dataset[centroides_indices]
        self.medias = centroides

        self.covarianzas = np.array([np.cov(dataset.T) for _ in range(self.K)])
        self.log_verosimilitud_previa = self.calcular_log_verosimilitud(dataset, self.pesos, self.medias, self.covarianzas)

    def ajustar(self, dataset):
        self.inicializar_parametros(dataset)
        self.log_verosimilitud_previa = -np.inf
        for _ in range(self.max_iteraciones):
            responsabilidades = self.etapa_e(dataset)
            self.etapa_m(dataset, responsabilidades)
            log_verosimilitud_actual = self.calcular_log_verosimilitud(dataset, self.pesos, self.medias, self.covarianzas)
        
            if np.abs(log_verosimilitud_actual - self.log_verosimilitud_previa) < self.tol:
                break
            self.log_verosimilitud_previa = log_verosimilitud_actual
    
    def etapa_e(self, dataset):
        responsabilidades = np.zeros((dataset.shape[0], self.K))

        probabilidades = np.zeros((dataset.shape[0], self.K))

        for k in range(self.K):
            norm = multivariate_normal(mean=self.medias[k], cov=self.covarianzas[k],allow_singular = True)
            probabilidades[:, k] = norm.pdf(dataset)
        
        responsabilidades = np.nan_to_num(responsabilidades, nan=1)
     
        for k in range(self.K):
           
            responsabilidades[:, k] = self.pesos[k] * probabilidades[:, k]
        for i in range(dataset.shape[0]):
            responsabilidades[i, :] /= np.sum(responsabilidades[i, :])
    
        return responsabilidades
        
    
    def etapa_m(self, dataset, responsabilidades):
        suma = responsabilidades.sum(axis=0)
        self.pesos = suma / len(dataset)
        self.medias = np.matmul(responsabilidades.T, dataset)
        self.medias /= suma[:, None]
        self.covarianzas = np.array([np.dot(responsabilidades[:, k] * (dataset - self.medias[k]).T, (dataset - self.medias[k])) / suma[k]
                                      for k in range(self.K)])

    def clusters(self, dataset):
        responsabilidades = self.etapa_e(dataset)
        etiquetas = np.argmax(responsabilidades, axis=1)
        return etiquetas