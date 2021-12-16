import numpy as np
import typing

from hipo_rank import Embeddings, Document


class IdentityClustering:
    def get_clusters(self, embeds: Embeddings, doc: Document) -> typing.Tuple[Embeddings, Document]:
        return embeds, doc  # no change

    def flatten(self, embeds: Embeddings, doc: Document):
        all_sentences = np.array([s for t in doc.sections for s in t.sentences], dtype=object)
        all_embeddings = np.concatenate([x.embeddings for x in embeds.sentence], axis=0)
        return all_embeddings, all_sentences
