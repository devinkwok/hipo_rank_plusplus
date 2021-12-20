import torch
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


    # from Arthur's code
class KMeansWithBestK:
    def __init__(self, 
                 n_clusters,
                 select_best_n_cluster: bool=False,
                 max_clusters: int=15,
                 range = (3,4),  # empirically determined from previous runs, saves time to only try these values
                 ):
        self.select_best_n_cluster = select_best_n_cluster
        self.max_clusters = max_clusters
        self.range = range
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings):
        embeddings = self._normalize(embeddings)
        if self.select_best_n_cluster:
            k = self._pick_optimal_k(embeddings)
        else:
            k = self.n_clusters
        km = KMeans(k)
        return km.fit_predict(embeddings)

    def _normalize(self, embeddings):
        mms = MinMaxScaler()
        mms.fit(embeddings)
        return mms.transform(embeddings)

    def _pick_optimal_k(self, normalized_embeddings):
        Sum_of_squared_distances = []
        m = min(self.max_clusters, normalized_embeddings.shape[0])
        K = range(1,m)
        models = []
        for k in K:
            if (k >= self.range[0] and k <= self.range[1]) or k == 1 or k == m-1:
                km = KMeans(k)
                km = km.fit(normalized_embeddings)
                models.append(km)
                Sum_of_squared_distances.append(km.inertia_)
            else:
                Sum_of_squared_distances.append(None)
        dists = [self._distance_lineAB_pointC(
            np.array([K[0],Sum_of_squared_distances[0]]),
            np.array([K[-1],Sum_of_squared_distances[-1]]),
            np.array([k,p])) if p is not None else 0 for (k,p) in zip(K,Sum_of_squared_distances)]
        return np.argmax(dists) + 1

    def _distance_lineAB_pointC(self, A,B,C):
        u = B-A
        v = np.array([u[1], -u[0]])
        return np.abs((A-C)@v)/np.linalg.norm(v)
