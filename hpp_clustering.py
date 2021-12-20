from hipo_rank.dataset_iterators.pubmed import PubmedDataset, save_to_file

from hipo_rank.embedders.w2v import W2VEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.bert import BertEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.undirected import Undirected
from hipo_rank.directions.order import OrderBased
from hipo_rank.directions.edge import EdgeBased

from hipo_rank.scorers.add import AddScorer

from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

from hipo_rank.clusterings.cluster import IdentityClustering, \
    UnsupervisedClustering, RandomClusteringAlgorithm, \
    remove_duplicates_from_doc, intrasection_similarity, section_stats
from hipo_rank.clusterings.kmeans import KMeansWithBestK
from hipo_rank.clusterings.spectral import SpectralWithCosineAffinity
from hipo_rank.clusterings.louvain import LouvainClustering
from sklearn.cluster import SpectralClustering

from pathlib import Path
import json
import time
import numpy as np
import torch
from tqdm import tqdm


DEBUG = False

REMOVE_DUPLICATE_SENTENCES = [True, False]
DATASETS = [
    ("pubmed_val", PubmedDataset, {"file_path": "data/pubmed-release/val.txt"}),
    ("pubmed_val_no_sections", PubmedDataset,
     {"file_path": "data/pubmed-release/val.txt", "no_sections": True}
     ),
]
EMBEDDERS = [
    ("rand_200", RandEmbedder, {"dim": 200}),
    ("bert", BertEmbedder,
     {"bert_config_path": "",
      "bert_model_path": "",
      "bert_tokenizer": "bert-base-cased",
      "bert_pretrained": "bert-base-cased",
      "cuda": not DEBUG,
      }
    ),
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("undirected", Undirected, {}),
    ("order", OrderBased, {}),
    ("edge", EdgeBased, {}),
    ("backloaded_edge", EdgeBased, {"u": 0.8}),
    ("frontloaded_edge", EdgeBased, {"u": 1.2}),
]
CLUSTERINGS = [
    ('none', IdentityClustering, {}),
    ('random', UnsupervisedClustering, {
            "clustering_algorithm": RandomClusteringAlgorithm,
            "clustering_args": {},
            "debug": DEBUG,
        }),
    ('louvain_clustering', UnsupervisedClustering, {
        "clustering_algorithm": LouvainClustering,
        "clustering_args": {},
        "debug": DEBUG,}   
    ),
    ('kmeanspickk', UnsupervisedClustering, {
            "clustering_algorithm": KMeansWithBestK,
            "clustering_args": {"select_best_n_cluster": True, "range": (3,4)},
            "debug": DEBUG,
        }),
    ('kmeanssectk', UnsupervisedClustering, {
            "clustering_algorithm": KMeansWithBestK,
            "clustering_args": {"select_best_n_cluster": False},
            "debug": DEBUG,
        }),
    ('spectralcos', UnsupervisedClustering, {
            "clustering_algorithm": SpectralWithCosineAffinity,
            "clustering_args": {"assign_labels": "kmeans"},
            "debug": DEBUG,
        }),
    ('spectralrbf', UnsupervisedClustering, {
            "clustering_algorithm": SpectralClustering,
            "clustering_args": {"affinity": "rbf", "assign_labels": "kmeans"},
            "debug": DEBUG,
        }),
]
SCORERS = [
    ("add_f=0.0_b=1.0_s=1.0", AddScorer, {}),
]

Summarizer = DefaultSummarizer()

experiment_time = int(time.time())
results_path = Path("results/test" if DEBUG else "results/hpp_clustering")
print('debug=', DEBUG)
for dataset_id, dataset, dataset_args in DATASETS:
    for no_duplicates in REMOVE_DUPLICATE_SENTENCES:
        print(f"remove duplicates {no_duplicates}")
        DataSet = dataset(**dataset_args)
        original_docs = list(DataSet)
        if no_duplicates:
            original_docs = [remove_duplicates_from_doc(d) for d in original_docs]
        if DEBUG:
            original_docs = original_docs[:5]
        for embedder_id, embedder, embedder_args in EMBEDDERS:
            Embedder = embedder(**embedder_args)
            print(f"embedding dataset {dataset_id} with {embedder_id}")
            original_embeds = [Embedder.get_embeddings(doc) for doc in tqdm(original_docs)]
            for clustering_id, clustering, clustering_args in CLUSTERINGS:
                print(f"clustering dataset {dataset_id} with {clustering_id}")
                Clustering = clustering(**clustering_args)
                embeds_and_docs = [Clustering.get_clusters(e, d) for e, d in tqdm(zip(original_embeds, original_docs))]
                embeds, docs = tuple(zip(*embeds_and_docs))
                for similarity_id, similarity, similarity_args in SIMILARITIES:
                    Similarity = similarity(**similarity_args)
                    print(f"calculating similarities with {similarity_id}")
                    sims = [Similarity.get_similarities(e) for e in embeds]
                    for direction_id, direction, direction_args in DIRECTIONS:
                        print(f"updating directions with {direction_id}")
                        Direction = direction(**direction_args)
                        sims = [Direction.update_directions(s) for s in sims]
                        for scorer_id, scorer, scorer_args in SCORERS:
                            Scorer = scorer(**scorer_args)
                            experiment = f"{dataset_id}-{no_duplicates}-{embedder_id}-{clustering_id}-{similarity_id}-{direction_id}-{scorer_id}"
                            experiment_path = results_path / experiment
                            try:
                                experiment_path.mkdir(parents=True)
                                print("running experiment: ", experiment)
                                results = []
                                references = []
                                summaries = []
                                for sim, doc in zip(sims, docs):
                                    scores = Scorer.get_scores(sim)
                                    try:
                                        summary = Summarizer.get_summary(doc, scores)
                                    except:
                                        print(sim)
                                        raise Exception
                                    results.append({
                                        "num_sects": len(doc.sections),
                                        "num_sents": sum([len(s.sentences) for s in doc.sections]),
                                        "summary": summary,
                                    })
                                    summaries.append([s[0] for s in summary])
                                    references.append([doc.reference])
                                rouge_result = evaluate_rouge(summaries, references)
                                (experiment_path / "rouge_results.json").write_text(json.dumps(rouge_result, indent=2))
                                (experiment_path / "summaries.json").write_text(json.dumps(results, indent=2))
                            except FileExistsError:
                                print(f"{experiment} already exists, skipping...")
                                pass
                    # save stats on similarity scores within each section
                    sect_stats = np.stack([section_stats(d) for d in docs], axis=1)
                    sim_stats = np.stack([intrasection_similarity(s) for s in sims], axis=1)
                    stats = np.concatenate([sect_stats, sim_stats], axis=0)
                    if DEBUG:
                        print(stats)
                    torch.save(stats, results_path / f'{dataset_id}-{no_duplicates}-{embedder_id}-{clustering_id}-{similarity_id}-stats.pt')
                # save clusters for reference
                save_to_file(results_path / f'{dataset_id}-{no_duplicates}-{embedder_id}-{clustering_id}-clusters.txt', docs)
