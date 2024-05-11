import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pickle
import re
import spacy

from functools import partial
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple


DATA_DIR = "../data"
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")

VOCAB_VECTORS_FP = os.path.join(DATA_DIR, "graphrag-vocab-vecs")
GRAPH_MATRIX_FP = os.path.join(DATA_DIR, "graphrag-adj-matrix")
VOCAB_DICT_FP = os.path.join(DATA_DIR, "graphrag-vocab-dict.pkl")

SIMILARITY_THRESHOLD_DETERMINED = True

logging.basicConfig(level=logging.INFO)


class ChunkJson:
    def __init__(self, chunk_fp):
        self.chunk_fp = chunk_fp

    def read(self):
        logging.info("reading {:s}".format(self.chunk_fp))
        with open(self.chunk_fp, "r", encoding="utf-8") as f:
            chunk_doc_json = json.loads(f.read())
        # print(json.dumps(chunk_doc_json, indent=2))
        page_content = chunk_doc_json["kwargs"]["page_content"]
        return page_content


def _noun_phrase_tokenizer(nlp: spacy.language.Language,
                           text: str
                           ) -> List[str]:
    np_tokens = []
    doc_text = nlp(text)
    for sent in doc_text.sents:
        doc_sent = nlp(sent.text)
        for noun_phrase in doc_sent.noun_chunks:
            doc_np = nlp(noun_phrase.text)
            np_lemma_toks = []
            for tok in doc_np:
                if tok.is_stop or tok.is_punct or tok.is_digit:
                    continue
                lemma = tok.lemma_
                lemma_r = re.sub(r"([^a-z ]+)", " ", lemma)
                lemma_r = re.sub(r"\s+", " ", lemma_r)
                lemma_r = lemma_r.strip()
                if len(lemma_r) > 0:
                    np_lemma_toks.append(lemma_r)
            np_tokens.append(" ".join(np_lemma_toks))
    return list(set(np_tokens))


def plot_vocabulary_count_distribution(sorted_vocab: List[Tuple[str, int]]):
    vocab_counts = [c for t, c in sorted_vocab]
    plt.bar(np.arange(len(vocab_counts)), vocab_counts)
    plt.title("Noun phrase vocabulary")
    plt.xlabel("tokens")
    plt.ylabel("counts")
    _ = plt.show()


def compute_percentiles(values: List[int], percentiles: List[int]) -> str:
    return np.percentile(np.array(values), percentiles)


def flatten_sim_matrix(sim_matrix: np.ndarray) -> np.ndarray:
    sim_val_x_idxs, sim_val_y_idxs = np.triu_indices(
        sim_matrix.shape[0], k=1)
    assert len(sim_val_x_idxs) == len(sim_val_y_idxs)
    sim_values = np.zeros(len(sim_val_x_idxs))
    for i, j in zip(sim_val_x_idxs, sim_val_y_idxs):
        sim_values[i] = sim_matrix[i, j]
    return sim_values


def compute_descriptive_statistics(
        values: np.ndarray[float]) -> List[float]:
    sim_values_min = np.min(sim_values)
    sim_values_max = np.max(sim_values)
    sim_values_mean = np.mean(sim_values)
    sim_values_sd = np.std(sim_values)
    return sim_values_min, sim_values_max, sim_values_mean, sim_values_sd


def compute_num_edges_at_different_thresholds(
        sim_values: np.ndarray[float],
        start: float, end: float, num: int,
        should_plot: bool = False) -> List[Tuple[float, int]]:
    num_edges = []
    thresholds = np.linspace(start, end, num)
    for threshold in thresholds:
        mask = sim_values > threshold
        count = np.sum(mask)
        num_edges.append(count)
    if should_plot:
        plt.plot(thresholds, num_edges)
        plt.xlabel("similarity threshold")
        plt.ylabel("#-edges")
        plt.title("Distribution of #-edges with similarity threshold")
        _ = plt.show()
    return thresholds, num_edges


def plot_similarity_heatmaps(sim_matrix: np.ndarray, thresholds: List[float]):
    plt.figure(figsize=(10, 10))
    for i, threshold in enumerate(thresholds):
        plt.subplot(3, 3, i+1)
        mask = sim_matrix > threshold
        mask.dtype = np.int8
        plt.imshow(mask, cmap="gray", interpolation="nearest")
        plt.title("threshold: {:.3f}".format(threshold))
    plt.tight_layout()
    _ = plt.show()


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    noun_phrase_tokenizer = partial(_noun_phrase_tokenizer, nlp)

    chunk_fps = [os.path.join(CHUNKS_DIR, x) for x in os.listdir(CHUNKS_DIR)]
    inputs = [ChunkJson(chunk_fp) for chunk_fp in chunk_fps]

    vectorizer = CountVectorizer(
        input="file", tokenizer=noun_phrase_tokenizer)
    docs = vectorizer.fit_transform(inputs)
    print("doc matrix shape:", docs.shape)

    sparse_sim_matrix = docs.T @ docs
    print("sparse sim matrix shape:", sparse_sim_matrix.shape)

    # graph the distribution
    vocab = vectorizer.vocabulary_
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    # plot_vocabulary_count_distribution(sorted_vocab)

    noun_phrases = [noun_phrase for noun_phrase, _ in sorted_vocab]
    counts = [count for _, count in sorted_vocab]
    # # just to see if restricting vocab size makes sens
    # # (it doesn't in this case)
    # print(compute_percentiles(counts, [75, 80, 90, 95, 99, 100]))
    # # [5703.75 6084.   6844.5  7224.75 7528.95 7605.  ]

    noun_phrase_vectors = encoder.encode(noun_phrases,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True,
                                         show_progress_bar=True)

    dense_sim_matrix = noun_phrase_vectors @ noun_phrase_vectors.T
    print("dense sim matrix shape:", dense_sim_matrix.shape)

    # overlay the two similarity matrices, that way edges represent 
    # similarities between noun phrases which co-occurs in the corpus 
    # at least once
    sim_matrix = np.multiply(sparse_sim_matrix.todense(),
                             dense_sim_matrix)

    if not SIMILARITY_THRESHOLD_DETERMINED:
        # useful for figuring out thresholds for sim_matrix
        # (based on this a good threshold may be 0.3)
        sim_values = flatten_sim_matrix(sim_matrix)
        sim_values_stats = compute_descriptive_statistics(sim_values)
        print("sim_values stats:", sim_values_stats)

        thresholds, num_edges = compute_num_edges_at_different_thresholds(
            sim_values, 0, sim_values_stats[1], 9, should_plot=True)
        print("thresholds:", thresholds)
        print("num_edges:", num_edges)
        # # thresholds: [0., 0.06867559, 0.13735119, 0.20602678, 0.27470237, 0.34337796, 0.41205356, 0.48072915, 0.54940474]
        # # num_edges: [7522, 6846, 5340, 3471, 1843, 865, 232, 22, 0]
        plot_similarity_heatmaps(sim_matrix, thresholds)
    else:
        threshold = 0.3
        thresholded_sim_matrix = sim_matrix > threshold
        thresholded_sim_matrix.dtype = np.int8

    np.save(VOCAB_VECTORS_FP, noun_phrase_vectors)
    np.save(GRAPH_MATRIX_FP, thresholded_sim_matrix)
    with open(VOCAB_DICT_FP, "wb") as fvocab_dict_pkl:
        pickle.dump(vocab, fvocab_dict_pkl)
