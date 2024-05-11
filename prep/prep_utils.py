import os
import re

from langchain_core.documents import Document
from llama_index.core.schema import TextNode
from typing import List


def extract_chapter_from_fn(chapter_fn):
    m = re.match(r"chapter-(\d+).txt", chapter_fn)
    if m is not None:
        return m.group(1)
    else:
        return None


def convert_llamaindex_textnodes_to_langchain_docs(
        nodes: List[TextNode]) -> List[Document]:
    docs = []
    for chunk_id, node in enumerate(nodes):
        chapter_id = extract_chapter_from_fn(
            os.path.basename(node.metadata.get("file_name")))
        metadata = node.metadata
        metadata["doc_id"] = node.id_
        metadata["chapter_id"] = chapter_id
        metadata["chunk_id"] = chunk_id + 1
        for key, values in node.relationships.items():
            if isinstance(values, list):
                rel_node_ids = [v.node_id for v in values]
                metadata["rel_{:s}".format(key.name)] = rel_node_ids
            else:
                metadata["rel_{:s}".format(key.name)] = [values.node_id]
        if node.embedding is not None:
            metadata["embedding"] = node.embedding
        doc = Document(
            id=node.id_,
            page_content=node.text,
            metadata=metadata
        )
        docs.append(doc)
    return docs


def write_langchain_docs_to_disk(docs: List[Document],
                                 output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for doc in docs:
        doc_fn = "chunk-{:s}-{:d}.json".format(
            doc.metadata["chapter_id"],
            doc.metadata["chunk_id"])
        doc_fp = os.path.join(output_dir, doc_fn)
        with open(doc_fp, "w", encoding="utf-8") as fdoc:
            fdoc.write(doc.json(indent=2))
    return
