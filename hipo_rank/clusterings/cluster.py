import numpy as np
import typing

from hipo_rank import Embeddings, Document, Section


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
