import boto3
import faiss
import networkx as nx
import numpy as np
import os
import pickle
import re
import spacy

from dotenv import find_dotenv, load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from operator import itemgetter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List

from chain_utils import (
    extract_questions, sample_question,
    read_template_from_file, parse_response
)
from my_retrievers import (
    create_base_vector_retriever,
    LexicalVectorSequenceRetriever
)


DATA_DIR = "../data"

VOCAB_VECTORS_FP = os.path.join(DATA_DIR, "graphrag-vocab-vecs.npy")
GRAPH_MATRIX_FP = os.path.join(DATA_DIR, "graphrag-adj-matrix.npy")
VOCAB_DICT_FP = os.path.join(DATA_DIR, "graphrag-vocab-dict.pkl")

CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
TFIDF_CHAP_DIR = os.path.join(DATA_DIR, "tfidf-chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")

GRAPHRAG_PROMPT_1_FP = "graph_rag_chain_1.prompt.txt"
GRAPHRAG_PROMPT_2_FP = "graph_rag_chain_2.prompt.txt"
GRAPHRAG_PROMPT_3_FP = "graph_rag_chain_3.prompt.txt"


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


class QQPair(BaseModel):
    question: str = Field(alias="question", description="input question")
    rewritten: str = Field(alias="rewritten", description="rephrased question")


class QAEval(BaseModel):
    question: str = Field(alias="question", description="the question provided")
    answer_1: str = Field(alias="answer_1", description="the first answer provided")
    answer_2: str = Field(alias="answer_2", description="the second answer provided")
    explanation: str = Field(alias="explanation", description="step by step reasoning")
    decision: str = Field(alias="decision", description="ANSWER_1 or ANSWER_2")


def extract_noun_phrases_from_query(query: str,
                                    nlp: spacy.language.Language
                                    ) -> List[str]:
    np_tokens = []
    doc_text = nlp(query)
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


def build_np_graph(adj_matrix_fp: str) -> nx.Graph:
    adj_matrix = np.load(adj_matrix_fp)
    np.fill_diagonal(adj_matrix, 0.0)
    sim_graph = nx.from_numpy_array(adj_matrix)
    print("sim_graph (#-nodes: {:d}, #-edges: {:d})".format(
        sim_graph.number_of_nodes(), sim_graph.number_of_edges()))
    return sim_graph


def build_vocabulary_search_index(vocab_vectors_fp: str) -> faiss.IndexIDMap:
    vocab_vectors = np.load(vocab_vectors_fp)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vocab_vectors.shape[1]))
    index.add_with_ids(vocab_vectors, np.arange(vocab_vectors.shape[0]))
    return index


def build_vocab_dicts(vocab_dict_fp: str):
    with open(VOCAB_DICT_FP, "rb") as fvocab_dict:
        vocab_dict = pickle.load(fvocab_dict)
    vocab_dict_r = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, vocab_dict_r


def extract_keywords_from_graph(question, nlp, index, sim_graph):
    query_nps = extract_noun_phrases_from_query(question, nlp)
    encoded_nps = encoder.encode(query_nps,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True,
                                 show_progress_bar=True)
    matched_nps = []
    for query_np, encoded_np in zip(query_nps, encoded_nps):
        matched_nps.append(query_np)
        dists, nearest_nbrs = index.search(encoded_np.reshape(1, -1), k=1)
        for dist, nnbr in zip(dists, nearest_nbrs):
            if dist[0] > 0.95:
                matched_np = vocab_dict_r[nnbr[0]]
                matched_nps.append(matched_np)
                for nbr in sim_graph.neighbors(nnbr[0]):
                    matched_nps.append(vocab_dict_r[nbr])
                break
    return matched_nps


def do_naive_rag(question: str,
                 retriever: BaseRetriever,
                 model: LLM) -> str:
    prompt_template = read_template_from_file(GRAPHRAG_PROMPT_1_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )
    chain = prompt | model | StrOutputParser()
    context = retriever.get_relevant_documents(question)
    response = chain.invoke({
        "question": question,
        "context": context
    })
    result = parse_response(response)
    qa_pair = QAPair(**result.value["qa_pair"])
    answer = qa_pair.answer
    return answer, context


def rewrite_question_using_graphrag(question: str,
                                    nlp: spacy.language.Language,
                                    index: faiss.IndexIDMap,
                                    sim_graph: nx.Graph,
                                    model: LLM) -> str:

    key_phrases = extract_keywords_from_graph(question, nlp, index, sim_graph)
    print("key phrases:", key_phrases)
    
    prompt_template = read_template_from_file(GRAPHRAG_PROMPT_2_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "phrases"]
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "question": question,
        "phrases": "\n".join(key_phrases)
    })
    result = parse_response(response)
    qq_pair = QQPair(**result.value["qq_pair"])
    rewritten_question = qq_pair.rewritten
    # print("rewritten question:", rewritten_question)
    return rewritten_question


def evaluate_answers(question: str,
                     context: str,
                     answer: str,
                     new_answer: str,
                     model: LLM) -> int:

    prompt_template = read_template_from_file(GRAPHRAG_PROMPT_3_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "answer_1", "answer_2"]
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "question": question,
        "answer_1": answer,
        "answer_2": new_answer
    })
    result = parse_response(response)
    qa_eval = QAEval(**result.value["qa_eval"])
    return qa_eval.decision


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    sim_graph = build_np_graph(GRAPH_MATRIX_FP)
    index = build_vocabulary_search_index(VOCAB_VECTORS_FP)
    vocab_dict, vocab_dict_r = build_vocab_dicts(VOCAB_DICT_FP)

    nlp = spacy.load("en_core_web_sm")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    tfidf_retriever = TFIDFRetriever.load_local(TFIDF_CHAP_DIR)
    vector_retriever = create_base_vector_retriever(CHROMA_DIR)
    retriever = LexicalVectorSequenceRetriever.create(
        tfidf_retriever, vector_retriever)

    boto3_bedrock = boto3.client("bedrock-runtime")
    model = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs={
            "temperature": 0.0,
            "max_tokens_to_sample": 1024,
            "stop_sequences": ["\n\nHuman"]
        },
        streaming=True
    )

    questions = extract_questions(CHAPTERS_DIR)
    sampled_questions = sample_question(questions, num_samples=3)

    for _, question in sampled_questions:
        print("question:", question)

        # do naive RAG
        answer, context = do_naive_rag(question, retriever, model)
        print("answer (naive RAG):", answer)

        # do graph RAG
        new_question = rewrite_question_using_graphrag(
            question, nlp, index, sim_graph, model)
        print("new question:", new_question)
        new_answer, _ = do_naive_rag(new_question, retriever, model)
        print("answer from Graph RAG:", new_answer)

        # evaluate
        context_page_contents = "\n".join([doc.page_content for doc in context])
        decision = evaluate_answers(
            question, context_page_contents, answer, new_answer, model)
        print("decision:", decision)
        print("---")
