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
    def __init__(self, use_cosine_similarity, **args):
        self.use_cosine_similarity = use_cosine_similarity
        if self.use_cosine_similarity:
            args["affinity"] = "precomputed"
        self.spectral = SpectralClustering(**args)

    def cos_similarity(embeddings: np.ndarray):
        torch.cosine_similarity(embeddings, dim=1)
        pass  #TODO

    def fit_predict(self, embeddings: np.ndarray):
        data = embeddings
        if self.use_cosine_similarity:
            data = self.cos_similarity(embeddings)
        return self.spectral.fit_predict(data)


class UnsupervisedClustering(IdentityClustering):
    def __init__(self,
                 clustering_algorithm=RandomClusteringAlgorithm,
                 clustering_args: dict={},
                 select_best_n_cluster: bool=False,
                 debug=False
              ):
        self.select_best_n_cluster = select_best_n_cluster
        self.clustering_algorithm = clustering_algorithm  # must implement fit_predict()
        self.clustering_args = clustering_args
        self.debug = debug

    def get_clusters(self,  embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        # flatten doc/embeds into array of sentences/embeds
        all_embeddings, all_sentences = self.flatten(embeds, doc)
        if len(all_embeddings) == 1:
            return embeds, doc  # do not cluster if only 1 sentence
        # get cluster labels from embeddings
        if self.select_best_n_cluster:
            n_cluster = 1  # TODO select best n_cluster
        else:
            n_cluster = min(len(all_embeddings), len(embeds.section))
        cluster_obj = self.clustering_algorithm(
            **self.clustering_args, n_clusters=n_cluster)
        try:
            cluster_labels = cluster_obj.fit_predict(all_embeddings)
        except:
            print(all_embeddings.shape)
            print(len(all_sentences))
            print(doc.reference)
            raise ValueError
        # create new Embeddings and Document object with sections based on cluster labels
        # generate new Embedding object
        cluster_ids = ["CLUSTER" + str(i) for i in range(n_cluster)]
        cluster_masks = np.stack([(cluster_labels == i) for i in range(n_cluster)], axis=0)
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
        #     similarities = self.cos_similarity(all_embeddings)
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
