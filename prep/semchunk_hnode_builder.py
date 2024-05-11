import glob
import numpy as np
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.schema import TransformComponent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from summarizer import Summarizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List

from prep_utils import (
    convert_llamaindex_textnodes_to_langchain_docs,
    write_langchain_docs_to_disk
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHUNKS_DIR = os.path.join(DATA_DIR, "llamaindex-chunks")
CHROMADB_PATH = os.path.join(DATA_DIR, "hier-chroma-db")


class SemanticChunkingSummaryNodesBuilder(TransformComponent):
    _approx_cluster_size: int = PrivateAttr()
    _extractive_summarizer: Summarizer = PrivateAttr()
    _abstractive_summarizer: pipeline = PrivateAttr()
    _embedding_generator: SentenceTransformer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._approx_cluster_size = kwargs.get("approx_cluster_size", 10)
        self._extractive_summarizer = Summarizer(
            kwargs.get("extractive_summarizer_model_name", "bert-base-uncased"))
        self._abstractive_summarizer = pipeline(
            "summarization", model=kwargs.get(
                "absractive_summarizer_model_name",
                "facebook/bart-large-cnn"))
        self._embedding_generator = kwargs.get("embedding_generator")

    def _compute_similarities(self, nodes: List[TextNode]) -> List[float]:
        prev_emb = None
        sims = []
        for node in nodes:
            if prev_emb is not None:
                sim = np.dot(np.array(node.embedding), prev_emb)
                sims.append(sim)
            prev_emb = np.array(node.embedding)
        return sims

    def _compute_threshold(self, sims: List[float]) -> float:
        pct_pt = 100.0 - (self._approx_cluster_size * 100 / len(sims))
        threshold = np.percentile(sims, pct_pt)
        return threshold
    
    def _cluster_nodes(self,
                       similarities: List[float],
                       threshold: float
                       ) -> List[List[int]]:
        clusters, cluster = [], []
        for idx, sim in enumerate(similarities):
            cluster.append(idx)
            if sim < threshold:
                clusters.append(cluster)
                cluster = []
        if len(cluster) > 0:
            clusters.append(cluster)
        return clusters
    
    def _do_extractive_summarization(self, texts: List[str]) -> List[str]:
        summaries = []
        compression_ratio = 1.0 / len(texts)
        for text in texts:
            summary = self._extractive_summarizer(
                body=text, ratio=compression_ratio)
            summaries.append(summary)
        return summaries

    def _do_abs_summarization(self, texts: List[str]) -> List[str]:
        input = "\n".join(["* " + text for text in texts])
        max_length = max([len(text) for text in texts])
        if max_length > 512:
            max_length = 512
        output = self._abstractive_summarizer(input, max_length=max_length)
        summary = output[0]["summary_text"]
        return summary

    def _summarize_cluster(self,
                           cluster: List[int],
                           nodes: List[TextNode]) -> TextNode:
        # inherit metadata from first child
        summary_node = TextNode(metadata=nodes[cluster[0]].metadata)
        texts = [nodes[idx].text for idx in cluster]
        ext_summaries = self._do_extractive_summarization(texts)
        abs_summary = self._do_abs_summarization(ext_summaries)
        # update the text with summary
        summary_node.text = abs_summary
        # set the children links for summary
        summary_node.relationships[NodeRelationship.CHILD] = [
            RelatedNodeInfo(node_id=nodes[idx].id_) for idx in cluster]
        # set the parent links to summary for children
        for idx in cluster:
            nodes[idx].relationships[NodeRelationship.PARENT] = \
                RelatedNodeInfo(node_id=summary_node.id_)
        return summary_node, nodes

    def _compute_summary_embeddings(self, nodes: List[TextNode]
                                    ) -> List[TextNode]:
        # identify summary nodes and extract texts
        summary_node_ids, summary_texts = [], []
        for node in nodes:
            if NodeRelationship.CHILD in node.relationships.keys():
                # indicates summary node
                summary_node_ids.append(node.id_)
                summary_texts.append(node.text)
        # compute embeddings for summary texts
        embeddings = self._embedding_generator.encode(
            summary_texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True)
        # lookup and update the embeddings for summary nodes
        node_to_embeddings = {node_id: emb.tolist()
                              for node_id, emb in zip(
                                  summary_node_ids, embeddings)}
        for node in nodes:
            if node.id_ in node_to_embeddings.keys():
                node.embedding = node_to_embeddings[node.id_]
        return nodes
      
    def __call__(self, nodes, **kwargs):
        sims = self._compute_similarities(nodes)
        threshold = self._compute_threshold(sims)
        clusters = self._cluster_nodes(sims, threshold)
        for cluster in clusters:
            if len(cluster) > 1:
                summary_node, nodes = self._summarize_cluster(cluster, nodes)
                nodes.append(summary_node)
        nodes = self._compute_summary_embeddings(nodes)
        return nodes


if __name__ == "__main__":

    text_splitter = SentenceSplitter(
        chunk_size=500, chunk_overlap=100, paragraph_separator="\n\n")
    embedding_generator = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        normalize=True)
    summary_node_builder = SemanticChunkingSummaryNodesBuilder(
        approx_cluster_size=10,
        extractive_summarizer_model_name="bert-base-uncased",
        absractive_summarizer_model_name="facebook/bart-large-cnn",
        embedding_generator=SentenceTransformer(
            "sentence-transformers/all-MiniLM-L12-v2"))
    pipeline_transformations = [
        text_splitter,
        embedding_generator,
        summary_node_builder
    ]
    ingestion_pipeline = IngestionPipeline(
        transformations=pipeline_transformations)

    for chapter_file in glob.glob(os.path.join(CHAPTERS_DIR, "chapter-*.txt")):
        docs = SimpleDirectoryReader(input_files=[chapter_file]).load_data()
        nodes = ingestion_pipeline.run(documents=docs)
        print(chapter_file, len(docs), len(nodes))
        langchain_docs = convert_llamaindex_textnodes_to_langchain_docs(nodes)
        write_langchain_docs_to_disk(langchain_docs, CHUNKS_DIR)
    