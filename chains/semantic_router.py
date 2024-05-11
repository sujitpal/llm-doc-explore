import joblib
import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import BedrockChat
from operator import itemgetter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from chain_utils import (
    extract_questions, sample_question,
    read_template_from_file, parse_response
)
from my_retrievers import create_base_vector_retriever


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
RAG_CHAIN_PROMPT_FP = "rag_chain.prompt.txt"

UMAP_MODEL_FP = os.path.join(DATA_DIR, "semantic_router_umap.joblib")
KMEANS_MODEL_FP = os.path.join(DATA_DIR, "semantic_router_kmeans.joblib")
CLUSTER_TO_CHAPTERS = {
    0: [3, 6, 9, 10],
    1: [1, 7, 8],
    2: [0, 2, 4, 5, 11]
}


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    umap_model = joblib.load(UMAP_MODEL_FP)
    kmeans_model = joblib.load(KMEANS_MODEL_FP)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    model = BedrockChat(
        model_id="anthropic.claude-v2",
        model_kwargs={
            "temperature": 0.0
        })

    questions = extract_questions(CHAPTERS_DIR)
    chapter, question = sample_question(questions)
    print("question (chapter: {:s}): {:s}".format(chapter, question))

    qvec = encoder.encode([question],
                          convert_to_numpy=True,
                          normalize_embeddings=True,
                          show_progress_bar=True)
    qvec_r = umap_model.transform(qvec)
    pred_cluster = kmeans_model.predict(qvec_r)[0]

    print("predicted cluster: {:d}")
    print("chapter_ids:", CLUSTER_TO_CHAPTERS[pred_cluster])
    
    vector_retriever = create_base_vector_retriever(CHROMA_DIR)
    # see https://github.com/langchain-ai/langchain/discussions/10537
    search_kwargs = {
        "k": 5,
        "filter": {
            "chapter": {
                "$in": CLUSTER_TO_CHAPTERS[pred_cluster]
            }
        }
    }
    vector_retriever.search_kwargs = search_kwargs
    print("search_kwargs:", search_kwargs)
    docs = vector_retriever.get_relevant_documents(question)

    prompt_template = read_template_from_file(RAG_CHAIN_PROMPT_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )

    chain = (
        {
            "context": itemgetter("question") | vector_retriever,
            "question": itemgetter("question")
        }
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": question
    })
    # print(response)
    result = parse_response(response)
    # print(result)
    qapair = result.value["qa_pair"]
    print("QAPair:", qapair)
