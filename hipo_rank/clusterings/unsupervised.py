import typing
import torch
import numpy as np

from hipo_rank import Document, Section, Embeddings, SentenceEmbeddings, SectionEmbedding, Similarities
from hipo_rank.clusterings.cluster import IdentityClustering
from hipo_rank.similarities.cos import CosSimilarity

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import MinMaxScaler


def print_sentence_summary(sentences: typing.List[str], ids: typing.List[int]=None):
    if ids is None:
        ids = [''] * len(sentences)
    MAX_PRINT_LEN = 60
    for id, sentence in zip(ids, sentences):
        print('\t', id, (sentence[:MAX_PRINT_LEN] + '...') if len(sentence) > MAX_PRINT_LEN else sentence)
    print('')


class RandomClusteringAlgorithm:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings: np.ndarray):
        return np.random.randint(self.n_clusters, size=len(embeddings))


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


class UnsupervisedClustering(IdentityClustering):
    def __init__(self,
                 clustering_algorithm=RandomClusteringAlgorithm,
                 clustering_args: dict={},
                 debug=False
              ):
        self.clustering_algorithm = clustering_algorithm  # must implement fit_predict()
        self.clustering_args = clustering_args
        self.debug = debug

    def _initialize_clustering(self, n_clusters: int):
        args = {**self.clustering_args}
        args["n_clusters"] = n_clusters
        cluster_obj = self.clustering_algorithm(**args)
        return cluster_obj

    def get_clusters(self,  embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        # flatten doc/embeds into array of sentences/embeds
        all_embeddings, all_sentences = self.flatten(embeds, doc)
        if len(all_embeddings) == 1:
            return embeds, doc  # do not cluster if only 1 sentence
        # get cluster labels from embeddings
        n_cluster = min(len(all_embeddings), len(embeds.section))
        cluster_obj = self._initialize_clustering(n_cluster)
        try:
            cluster_labels = cluster_obj.fit_predict(all_embeddings)
        except:
            print(all_embeddings.shape)
            print(len(all_sentences))
            print(doc.reference)
            raise ValueError
        # create new Embeddings and Document object with sections based on cluster labels
        clusters = list(set(cluster_labels))  # omit empty clusters
        # generate new Embedding object
        cluster_ids = ["CLUSTER" + str(i) for i in range(len(clusters))]
        cluster_masks = np.stack([(cluster_labels == i) for i in clusters], axis=0)
        embeds_by_cluster = [all_embeddings[m, :] for m in cluster_masks]
        sentence_embeddings = [SentenceEmbeddings(id=i, embeddings=e)
                               for i, e in zip(cluster_ids, embeds_by_cluster)]
        # sum section embeddings over clusters
        section_embeddings = [SectionEmbedding(id=i, embedding=np.mean(e, axis=0))
                              for i, e in zip(cluster_ids, embeds_by_cluster)]
        embeds_obj = Embeddings(sentence=sentence_embeddings, section=section_embeddings)
        # generate new Document object
        section_sentences = [all_sentences[m] for m in cluster_masks]
        doc_sections = [Section(id=i, sentences=list(s))
                        for i, s in zip(cluster_ids, section_sentences)]
        doc_obj = Document(sections=doc_sections, reference=doc.reference)
        # sanity check
        if self.debug:
            for id, sentences in zip(cluster_ids, section_sentences):
                print(id, 'n_sentences =', len(sentences))
                print_sentence_summary(sentences)
        return embeds_obj, doc_obj

def section_stats(doc: Document):
    n_sections = len(doc.sections)
    n_sentences = np.array([len(s.sentences) for s in doc.sections])
    return [n_sections, np.mean(n_sentences), np.min(n_sentences), np.max(n_sentences)]

