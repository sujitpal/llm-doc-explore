# NOTE: this is not a llamaindex component, it takes the cluster output
# produced using raptor_clustering.py and applies it to the cluster JSON
# files produced by semchunk_hnode_builder.py. Ideally, we would save the 
# GMM trained and use its predictions on the re-generated chunks in a new
# IngestionPipeline that substitutes the semantic chunking clusterer with 
# the RAPTOR clusterer.

import datetime
import json
import os
import shutil
import uuid

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from summarizer import Summarizer
from transformers import pipeline
from typing import Any, Dict, List

from prep_utils import convert_llamaindex_textnodes_to_langchain_docs


DATA_DIR = "../data"
INPUT_CHUNKS_DIR = os.path.join(DATA_DIR, "llamaindex-chunks")
OUTPUT_CHUNKS_DIR = os.path.join(DATA_DIR, "llamaindex-chunks-raptor")
CLUSTER_PREDS_FP = os.path.join(DATA_DIR, "gmm_cluster_preds.tsv")


def convert_chunk_to_doc(chunk_dir: str, chunk_fn: str) -> dict:
    chunk_fp = os.path.join(chunk_dir, chunk_fn)
    with open(chunk_fp, "r", encoding="utf-8") as f:
        chunk = json.load(f)
    return Document(**chunk)


def do_extractive_summarization(summarizer, texts: List[str]) -> List[str]:
    summaries = []
    compression_ratio = 1.0 / len(texts)
    for text in texts:
        summary = summarizer(body=text, ratio=compression_ratio)
        summaries.append(summary)
    return summaries


def do_abs_summarization(summarizer, texts: List[str]) -> List[str]:
    input = "\n".join(["* " + text for text in texts])
    if len(input) > 1024:
        input = input[:1024]
    max_length = max([len(text) for text in texts])
    if max_length > 512:
        max_length = 512
    output = summarizer(input, max_length=max_length)
    summary = output[0]["summary_text"]
    return summary


def summarize_chunks(chunk_docs: List[Document],
                     extractive_summarizer,
                     abstractive_summarizer) -> str:
    chunk_texts = [doc.page_content for doc in chunk_docs]
    ext_summaries = do_extractive_summarization(extractive_summarizer,
                                                chunk_texts)
    abs_summary = do_abs_summarization(abstractive_summarizer, ext_summaries)
    return abs_summary


def generate_summary_metadata(children: List[Document],
                              summary_fp: str) -> Dict[str, Any]:
    summary_meta = children[0].metadata.copy()
    summary_meta["file_path"] = summary_fp
    summary_meta["file_name"] = summary_fp
    summary_meta["doc_id"] = str(uuid.uuid4())
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    summary_meta["creation_date"] = current_date
    summary_meta["last_modified_date"] = current_date
    summary_meta["rel_CHILD"] = [child.metadata["doc_id"]
                                 for child in children]
    summary_meta["chapter_id"] = list(set([child.metadata["chapter_id"]
                                           for child in children]))
    summary_meta["chunk_id"] = list(set([child.metadata["chunk_id"]
                                         for child in children]))
    for key in ["rel_PREVIOUS", "rel_NEXT", "embedding"]:
        if key in summary_meta.keys():
            del summary_meta[key]
    return summary_meta


# first create a dict of cluster labels to list of files
label_to_files = {}
with open(CLUSTER_PREDS_FP, "r", encoding="utf-8") as f:
    for line in f:
        chunk_fn, pred = line.strip().split("\t")
        pred = int(pred)
        if pred not in label_to_files.keys():
            label_to_files[pred] = [chunk_fn]
        else:
            label_to_files[pred].append(chunk_fn)
# remove clusters with only one document
filt_dict = {k: v for k, v in label_to_files.items() if len(v) > 1}            
label_to_files = filt_dict

shutil.rmtree(OUTPUT_CHUNKS_DIR, ignore_errors=True)
os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)
summary_fns = []
for label, chunk_fns in label_to_files.items():
    # create summary parent node using LlamaIndex
    summary_fn = "summary-raptor-{:d}.json".format(label)
    summary_fp = os.path.join(OUTPUT_CHUNKS_DIR, summary_fn)
    chunk_docs = [convert_chunk_to_doc(INPUT_CHUNKS_DIR, chunk_fn)
                  for chunk_fn in chunk_fns]
    
    extractive_summarizer = Summarizer("bert-base-uncased")
    abstractive_summarizer = pipeline("summarization",
                                      "facebook/bart-large-cnn")
    summary_doc = Document(
        page_content=summarize_chunks(chunk_docs, 
                                      extractive_summarizer,
                                      abstractive_summarizer),
        metadata=generate_summary_metadata(chunk_docs, summary_fp)
    )
    with open(summary_fp, "w", encoding="utf-8") as f:
        f.write(summary_doc.json(indent=2))
    summary_fns.append(summary_fn)
    # update rel_PARENT in children docs before writing to output
    for chunk in chunk_docs:
        chunk.metadata["rel_PARENT"] = [summary_doc.metadata["doc_id"]]
        chunk_fn = "chunk-{:s}-{:d}.json".format(
            chunk.metadata["chapter_id"],
            chunk.metadata["chunk_id"])
        chunk_fp = os.path.join(OUTPUT_CHUNKS_DIR, chunk_fn)
        with open(chunk_fp, "w", encoding="utf-8") as f:
            f.write(chunk.json(indent=2))

# compute embeddings for summary nodes and write back
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
summary_docs = [convert_chunk_to_doc(OUTPUT_CHUNKS_DIR, summary_fn)
                for summary_fn in summary_fns]
summary_texts = [doc.page_content for doc in summary_docs]
embeddings = encoder.encode(summary_texts, 
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=True)
for summary_fns, emb in zip(summary_fn, embeddings):
    summary_fp = os.path.join(OUTPUT_CHUNKS_DIR, summary_fn)
    with open(summary_fp, "r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["metadata"]["embedding"] = emb.tolist()
    with open(summary_fp, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
