import boto3
import argparse
import os

from dotenv import find_dotenv, load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain_community.retrievers import TFIDFRetriever

from chain_utils import (
    extract_questions, sample_question,
)
from my_retrievers import (
    RetrieverType,
    create_base_vector_retriever,
    LexicalVectorSequenceRetriever,
    LexicalVectorParallelRRFRetriever,
    WebSearchRetriever,
    HydeRetriever,
    RAGFusionRetriever,
    ParentDocumentRetriever
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
TFIDF_CHAP_DIR = os.path.join(DATA_DIR, "tfidf-chapters")
TFIDF_CHUNK_DIR = os.path.join(DATA_DIR, "tfidf-chunks")


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="vector",
                        help="choose a retriever",
                        choices=[e.value for e in RetrieverType])
    args = parser.parse_args()

    questions = extract_questions(CHAPTERS_DIR)
    chapter, question = sample_question(questions)
    print("question (chapter: {:s}): {:s}".format(chapter, question))

    boto3_bedrock = boto3.client("bedrock-runtime")
    model = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs={
            "temperature": 0.3,
            "max_tokens_to_sample": 1024,
            "stop_sequences": ["\n\nHuman"]
        }
    )

    retriever_type = RetrieverType(args.retriever)
    retriever = None
    match retriever_type:
        case RetrieverType.VECTOR:
            retriever = create_base_vector_retriever(CHROMA_DIR)
        case RetrieverType.LEX_VEC_SEQ:
            tfidf_retriever = TFIDFRetriever.load_local(TFIDF_CHAP_DIR)
            vector_retriever = create_base_vector_retriever(CHROMA_DIR)
            retriever = LexicalVectorSequenceRetriever.create(
                tfidf_retriever, vector_retriever)
        case RetrieverType.LEX_VEC_PAR:
            tfidf_retriever = TFIDFRetriever.load_local(TFIDF_CHUNK_DIR)
            vector_retriever = create_base_vector_retriever(CHROMA_DIR)
            retriever = LexicalVectorParallelRRFRetriever.create(
                tfidf_retriever, vector_retriever)
        case RetrieverType.WEB:
            retriever = WebSearchRetriever.create(model)
        case RetrieverType.HYDE:
            vector_retriever = create_base_vector_retriever(CHROMA_DIR)
            retriever = HydeRetriever.create(model, vector_retriever)
        case RetrieverType.RAG_FUSION:
            tfidf_retriever = TFIDFRetriever.load_local(TFIDF_CHAP_DIR)
            vector_retriever = create_base_vector_retriever(CHROMA_DIR)
            retriever = RAGFusionRetriever.create(
                model, tfidf_retriever, vector_retriever)
        case RetrieverType.PARENT:
            vector_retriever = create_base_vector_retriever(CHROMA_DIR)
            parent_retriever = TFIDFRetriever.load_local(TFIDF_CHAP_DIR)
            retriever = ParentDocumentRetriever.create(
                vector_retriever, parent_retriever)
        case _:
            raise ValueError("Invalid choice, should define in RetrieverType")

    if retriever is not None:
        docs = retriever.get_relevant_documents(question)
        for doc in docs:
            print(doc)
