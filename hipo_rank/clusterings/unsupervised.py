import typing
import torch
import numpy as np

from hipo_rank import Document, Section, Embeddings, SentenceEmbeddings, SectionEmbedding, Similarities
from hipo_rank.clusterings.cluster import IdentityClustering
from hipo_rank.similarities.cos import CosSimilarity

from sklearn.cluster import SpectralClustering


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


class UnsupervisedClustering(IdentityClustering):
    def __init__(self,
                 clustering_algorithm=RandomClusteringAlgorithm,
                 clustering_args: dict={},
                 select_best_n_cluster: bool=False,
                 max_n_clustering_to_try: int=16,
                 debug=False
              ):
        self.select_best_n_cluster = select_best_n_cluster
        self.clustering_algorithm = clustering_algorithm  # must implement fit_predict()
        self.clustering_args = clustering_args
        self.debug = debug

    def _initialize_clustering(self, n_clusters: int):
        args = {**self.clustering_args}
        args["n_clusters"] = n_clusters
        cluster_obj = self.clustering_algorithm(**args)
        return cluster_obj

    # from Arthur's code
    def _pick_optimal_k(self, embeddings):
        Sum_of_squared_distances = []
        for n in range(1, min(len(embeddings), self.max_clusters)):
            clustering = self._initialize_clustering(n)
            model = clustering.fit(embeddings)
            #FIXME only works for k-means
            Sum_of_squared_distances.append(model.inertia_)
        dists = [self._distance_lineAB_pointC(
                np.array([K[0],Sum_of_squared_distances[0]]),
                np.array([K[-1],Sum_of_squared_distances[-1]]),
                np.array([k,p]))
            for (k,p) in zip(K,Sum_of_squared_distances)]
        return np.argmax(dists) + 1

    def get_clusters(self,  embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        # flatten doc/embeds into array of sentences/embeds
        all_embeddings, all_sentences = self.flatten(embeds, doc)
        if len(all_embeddings) == 1:
            return embeds, doc  # do not cluster if only 1 sentence
        # get cluster labels from embeddings
        if self.select_best_n_cluster:
            n_cluster = self._pick_optimal_k(all_embeddings)
        else:
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
            similarities = self.cos_similarity(all_embeddings)
            print(np.min(similarities), np.max(similarities), np.mean(similarities))
            # assert np.all(np.sum(cluster_masks, axis=0) == 1), cluster_masks
            # # TODO plot clusterings
            # print(all_embeddings.shape, cluster_labels.shape, cluster_labels)
            # n_sentences = sum([len(x.sentences) for x in doc.sections])
            # n_new_sentences = sum([len(x.sentences) for x in doc_obj.sections])
            # assert n_sentences == n_new_sentences, (n_sentences, n_new_sentences)
            for id, sentences in zip(cluster_ids, section_sentences):
                print(id, 'n_sentences =', len(sentences))
                print_sentence_summary(sentences)
        return embeds_obj, doc_obj

def section_stats(doc: Document):
    n_sections = len(doc.sections)
    n_sentences = np.array([len(s.sentences) for s in doc.sections])
    return [n_sections, np.mean(n_sentences), np.min(n_sentences), np.max(n_sentences)]

