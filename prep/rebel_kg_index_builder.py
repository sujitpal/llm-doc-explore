import faiss
import json
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle

from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, save_npz


DATA_DIR = "../data"
REBEL_CHUNKS_DIR = os.path.join(DATA_DIR, "rebel-chunks")

FAISS_INDEX_FP = os.path.join(DATA_DIR, "rebel-faiss-index.bin")
GRAPH_MATRIX_FP = os.path.join(DATA_DIR, "rebel-adj-matrix")
VOCAB_DICT_FP = os.path.join(DATA_DIR, "rebel-vocab-dict.pkl")

if __name__ == "__main__":
    # extract triples from enriched chunks
    kg_triples = []
    for chunk_fn in os.listdir(REBEL_CHUNKS_DIR):
        chunk_fp = os.path.join(REBEL_CHUNKS_DIR, chunk_fn)
        with open(chunk_fp, "r") as f:
            chunk_json = json.loads(f.read())
            if "triplets" in chunk_json["metadata"].keys():
                triplets = chunk_json["metadata"]["triplets"]
                kg_triples.extend(triplets)

    # construct adjacency matrix from list of triples. Because we want to 
    # navigate two neighbors out including the relation predicate (type)
    # we will reify the graph and treat each triple as 3 nodes. So each
    # triple {"head": "A", "type": "R", "tail": "B"} will be represented as
    # 3 nodes: (A)->(R)->(B).
    unique_ents, subj_nodes, obj_nodes = set(), set(), set()
    for triple in kg_triples:
        unique_ents.add(triple["head"])
        unique_ents.add(triple["type"])
        unique_ents.add(triple["tail"])
        subj_nodes.update([triple["head"], triple["type"]])
        obj_nodes.update([triple["type"], triple["tail"]])

    # token to id and id to token mappings (saving only one for query side)
    id2token, token2id = {}, {}
    unique_ents = sorted(list(unique_ents))
    for kg_node_id, kg_node_name in enumerate(unique_ents):
        id2token[kg_node_id] = kg_node_name
        token2id[kg_node_name] = kg_node_id

    with open(VOCAB_DICT_FP, "wb") as f:
        pickle.dump(id2token, f)
                
    # sparse adjacency matrix for graph, saving as CSR matrix
    row_idxs, col_idxs, data = [], [], []
    for triple in kg_triples:
        row_idxs.append(token2id[triple["head"]])
        col_idxs.append(token2id[triple["type"]])
        data.append(1)
        row_idxs.append(token2id[triple["type"]])
        col_idxs.append(token2id[triple["tail"]])
        data.append(1)

    num_ents = len(unique_ents)
    adj_matrix = csr_matrix((data, (row_idxs, col_idxs)),
                            shape=(num_ents, num_ents))
    print("adj_matrix shape:", adj_matrix.shape)

    save_npz(GRAPH_MATRIX_FP, adj_matrix)

    # FAISS index for vector lookup of nodes
    # NOTE: seems to be some conflict with my environment, I get seg fault
    # here. If I move the FAISS import to after the encode call, then all 
    # is well. 
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    kg_node_vectors = encoder.encode(unique_ents,
                                     convert_to_numpy=True,
                                     normalize_embeddings=True,
                                     show_progress_bar=True)

    index = faiss.IndexIDMap(faiss.IndexFlatIP(kg_node_vectors.shape[1]))
    kg_node_ids = [token2id[ent] for ent in unique_ents]
    index.add_with_ids(kg_node_vectors, kg_node_ids)
    faiss.write_index(index, FAISS_INDEX_FP)

    # construct the graph from adj matrix and draw it
    G = nx.from_scipy_sparse_array(adj_matrix)
    nx.draw(G, with_labels=False)
    plt.title("Knowledge Graph produced from REBEL triples")
    _ = plt.show()
