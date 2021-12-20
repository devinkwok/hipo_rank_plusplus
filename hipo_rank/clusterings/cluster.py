import numpy as np
import typing

from hipo_rank import Embeddings, Document, Section, \
    Similarities, SentenceEmbeddings, SectionEmbedding


def print_sentence_summary(sentences: typing.List[str], ids: typing.List[int]=None):
    if ids is None:
        ids = [''] * len(sentences)
    MAX_PRINT_LEN = 60
    for id, sentence in zip(ids, sentences):
        print('\t', id, (sentence[:MAX_PRINT_LEN] + '...') if len(sentence) > MAX_PRINT_LEN else sentence)
    print('')


# from Arthur's code
def remove_duplicates_from_doc(doc: Document):
    sentence_set = set()
    new_sections = []
    for sec in doc.sections:
        sec_sentences = []
        for s in sec.sentences:
            if s not in sentence_set:
                sentence_set.add(s)
                sec_sentences.append(s)
        if len(sec_sentences) > 0:  # omit empty sections
            new_sections.append(Section(id=sec.id, sentences=sec_sentences))
    return Document(sections=new_sections, reference=doc.reference)


def section_stats(doc: Document):
    n_sections = len(doc.sections)
    n_sentences = np.array([len(s.sentences) for s in doc.sections])
    return [n_sections, np.mean(n_sentences), np.min(n_sentences), np.max(n_sentences)]


def intrasection_similarity(similarities: Similarities):
    means, stds, mins, maxes = [], [], [], []
    for section in similarities.sent_to_sent:
        sims = section.similarities
        if len(sims) > 0:
            means += [np.mean(sims)]
            stds += [np.std(sims)]
            mins += [np.min(sims)]
            maxes += [np.max(sims)]
    means = np.array(means)
    stds = np.array(stds)
    mins = np.array(mins)
    maxes = np.array(maxes)
    if len(means > 0):
        return [np.mean(means), np.min(means), np.max(means), np.mean(stds), np.min(stds), np.max(stds), np.mean(mins), np.min(mins), np.mean(maxes), np.max(maxes)]
    else:
        return [np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.), np.array(1.)]


class IdentityClustering:
    def get_clusters(self, embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        return embeds, doc  # no change

    def flatten(self, embeds: Embeddings, doc: Document):
        all_sentences = np.array([s for t in doc.sections for s in t.sentences], dtype=object)
        all_embeddings = np.concatenate([x.embeddings for x in embeds.sentence], axis=0)
        return all_embeddings, all_sentences


class RandomClusteringAlgorithm:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings: np.ndarray):
        return np.random.randint(self.n_clusters, size=len(embeddings))


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
