import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import re
import spacy

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
from umap import UMAP
from sklearn.cluster import KMeans


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
SUMMARY_FP = os.path.join(DATA_DIR, "chapter_summaries.jsonl")
UMAP_MODEL_FP = os.path.join(DATA_DIR, "semantic_router_umap.joblib")
KMEANS_MODEL_FP = os.path.join(DATA_DIR, "semantic_router_kmeans.joblib")


def extract_id_from_name(chapter_fn: str):
    m = re.match(r"chapter-(\d+).txt", chapter_fn)
    if m is not None:
        return m.group(1)
    else:
        return None


# adapted from: 
# https://medium.com/analytics-vidhya/text-summarization-in-python-using-extractive-method-including-end-to-end-implementation-2688b3fd1c8c
def rerank_sentences(chapter_fp: str,
                     nlp: spacy.language.Language,
                     encoder: SentenceTransformer) -> List[str]:
    # read chapter text
    with open(chapter_fp, "r", encoding="utf-8") as fin:
        text = fin.read()
    # split into sentences
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentence = sent.text
        if len(sentence.strip()) == 0:
            continue
        sentences.append(sentence)
    # vectorize sentences
    sent_vecs = encoder.encode(sentences,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=True)
    # construct similarity matrix (cosine similarity)
    sim_matrix = sent_vecs @ sent_vecs.T
    # convert to networkx graph
    sim_graph = nx.from_numpy_array(sim_matrix)
    # call pagerank (textrank)
    scores = nx.pagerank(sim_graph, max_iter=500, tol=1.0e-3)
    # rerank sentences
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True)
    return ranked_sentences


def summarize_chapters(summary_jsonl_fp, top_tokens,
                       nlp, encoder):
    if os.path.exists(summary_jsonl_fp):
        return
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=top_tokens,
        chunk_overlap=top_tokens // 10,
        length_function=len,
        is_separator_regex=False,
    )

    f_summ = open(summary_jsonl_fp, "w", encoding="utf-8")
    
    for chapter_fn in os.listdir(CHAPTERS_DIR):
        if not chapter_fn.startswith("chapter-"):
            continue
        chap_id = extract_id_from_name(chapter_fn)
        if chap_id is None:
            continue
        chapter_fp = os.path.join(CHAPTERS_DIR, chapter_fn)
        print("Processing {:s} (chapter {:s})...".format(chapter_fn, chap_id))
        reranked_sentences = rerank_sentences(chapter_fp, nlp, encoder)
        reranked_text = " ".join([s for _, s in reranked_sentences])
        reranked_text = re.sub(r"\s+", " ", reranked_text)
        reranked_doc = Document(page_content=reranked_text)
        chunks = text_splitter.split_documents([reranked_doc])
        print("--- chapter {:s} ---".format(chap_id))
        print(chunks[0])
        f_summ.write(json.dumps({
            "chapter": chap_id,
            "content": chunks[0].page_content
        }) + "\n")

    f_summ.close()


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    summarize_chapters(SUMMARY_FP, 10_000, nlp, encoder)

    cids, contents = [], []
    with open(SUMMARY_FP, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line.strip())
            chapter = rec["chapter"]
            content = rec["content"]
            contents.append(content)
            cids.append(chapter)

    vectors = encoder.encode(contents,
                             convert_to_numpy=True,
                             normalize_embeddings=True,
                             show_progress_bar=True)
    print(vectors.shape)

    # dimensionality reduction
    umap_model = UMAP(random_state=0)
    reduced_vectors = umap_model.fit_transform(vectors)
    print(reduced_vectors.shape)

    # clustering
    n_clusters = 3
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    kmeans_model.fit(reduced_vectors)
    labels = kmeans_model.labels_

    # visualization
    colors = ["red", "blue", "green"]
    chapter_clusters = []
    for cid in np.arange(n_clusters):
        x_sub = [i for i, x in enumerate(labels) if x == cid]
        chapter_clusters.append(x_sub)
        reduced_vectors_sub = reduced_vectors[x_sub]
        plt.scatter(reduced_vectors_sub[:, 0],
                    reduced_vectors_sub[:, 1], 
                    color=colors[cid])
        for i in x_sub:
            plt.annotate(cids[i], (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    _ = plt.show()

    joblib.dump(umap_model, UMAP_MODEL_FP)
    joblib.dump(kmeans_model, KMEANS_MODEL_FP)

    print(chapter_clusters)
    # [[3, 6, 9, 10], [1, 7, 8], [0, 2, 4, 5, 11]]

