import numpy as np
import typing

from hipo_rank import Embeddings, Document, Section, Similarities


class IdentityClustering:
    def get_clusters(self, embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        return embeds, doc  # no change

    def flatten(self, embeds: Embeddings, doc: Document):
        all_sentences = np.array([s for t in doc.sections for s in t.sentences], dtype=object)
        all_embeddings = np.concatenate([x.embeddings for x in embeds.sentence], axis=0)
        return all_embeddings, all_sentences


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
