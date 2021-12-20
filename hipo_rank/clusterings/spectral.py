import torch
import numpy as np

from sklearn.cluster import SpectralClustering


class SpectralWithCosineAffinity:
    def __init__(self, **args):
        args["affinity"] = "precomputed"
        self.spectral = SpectralClustering(**args)

    def sim_matrix(self, embeddings: np.ndarray):
        assert len(embeddings.shape) == 2, embeddings.shape
        embeddings = torch.tensor(embeddings)
        a = embeddings.reshape(*embeddings.shape, 1)
        x, y = torch.broadcast_tensors(a, embeddings.T)
        matrix = torch.cosine_similarity(x, y, dim=1)
        assert matrix.shape == (len(embeddings), len(embeddings)), matrix.shape
        return matrix.numpy()

    def fit_predict(self, embeddings: np.ndarray):
        similarities = self.sim_matrix(embeddings)
        return self.spectral.fit_predict(similarities)
