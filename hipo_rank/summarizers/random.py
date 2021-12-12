import numpy as np
from hipo_rank import Scores, Document, Summary


class RandomSummarizer:
    def __init__(self, num_words: int = 200, stay_under_num_words: bool = False):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document,
                    sorted_scores: Scores = None) -> Summary:
        sentences = [(i, j, s) for i, t in enumerate(doc.sections) for j, s in enumerate(t.sentences)]
        random_scores = np.arange(len(sentences))
        np.random.shuffle(random_scores)
        num_words = 0
        summary = []
        i = 0
        while True:
            sect_idx, local_idx, sentence = sentences[random_scores[i]]
            num_words += len(sentence.split())
            if self.stay_under_num_words and num_words > self.num_words:
                break
            summary.append((sentence, random_scores[i].item(), sect_idx, local_idx, random_scores[i].item()))
            i += 1
            if num_words >= self.num_words:
                break
            if i >= len(random_scores):
                break
        return summary
