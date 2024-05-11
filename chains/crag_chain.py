# Corrective RAG (CRAG) -- https://arxiv.org/abs/2401.15884
# env: apollo
import asyncio
import boto3
import langchain
import os

from dotenv import find_dotenv, load_dotenv
from enum import Enum
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import List

from chain_utils import (
    extract_questions, sample_question,
    parse_response, read_template_from_file
)
from my_retrievers import (
    create_base_vector_retriever,
    LexicalVectorSequenceRetriever,
    WebSearchRetriever
)


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
TFIDF_CHAP_DIR = os.path.join(DATA_DIR, "tfidf-chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
CRAG_PROMPT_1_FP = "crag_chain_1.prompt.txt"
CRAG_PROMPT_2_FP = "crag_chain_2.prompt.txt"

UPPER_CONFIDENCE_THRESHOLD = 8
LOWER_CONFIDENCE_THRESHOLD = 4


class ContextQuality(Enum):
    CORRECT = 0
    AMBIGUOUS = 1
    INCORRECT = 2
    

class QCEval(BaseModel):
    question: str = Field(alias="question", description="the question asked")
    context: str = Field(alias="context", description="text of context provided")
    grade: int = Field(alias="the relevance grade provided (1-10)")
    explanation: str = Field(alias="explanation",
                             description="justification for grade")


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


async def evaluate_contexts(question: str,
                            docs: List[Document],
                            chain: Runnable):
    tasks = []
    for doc in docs:
        tasks.append(chain.ainvoke({
            "question": question,
            "context": doc.page_content
        }))
    responses = await asyncio.gather(*tasks)
    scores = []
    for response in responses:
        result = parse_response(response)
        print("---")
        print(result)
        scores.append(int(result.value["qc_eval"]["grade"]))
    return scores


def compute_context_quality(scores: List[int],
                            lower_threshold: int,
                            upper_threshold: int) -> ContextQuality:
    is_correct = len([score for score in scores 
                      if score > upper_threshold]) > 0
    if is_correct:
        return ContextQuality.CORRECT
    is_incorrect = len([score for score in scores
                        if score < lower_threshold]) == len(scores)
    if is_incorrect:
        return ContextQuality.INCORRECT
    return ContextQuality.AMBIGUOUS


async def pipeline():

    _ = load_dotenv(find_dotenv())

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

    web_retriever = WebSearchRetriever.create(model)

    questions = extract_questions(CHAPTERS_DIR)
    _, question = sample_question(questions)
    print("question:", question)

    langchain.debug = False
    
    # Phase 1: get context from retriever
    docs = retriever.get_relevant_documents(question)

    # Phase 2: do quality check for each question-context pair
    prompt_template = read_template_from_file(CRAG_PROMPT_1_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = prompt | model | StrOutputParser()

    scores = await evaluate_contexts(question, docs, chain)

    # Phase 3: compute quality and classify as CORRECT, AMBIGUOUS,
    # or INCORRECT
    quality = compute_context_quality(
        scores, LOWER_CONFIDENCE_THRESHOLD, UPPER_CONFIDENCE_THRESHOLD)
    print("quality:", quality)

    match (quality):
        case ContextQuality.CORRECT:
            pass
        case ContextQuality.AMBIGUOUS:
            # supplement with web_docs
            web_docs = web_retriever.get_relevant_documents(question)
            docs = docs.extend(web_docs)
        case ContextQuality.INCORRECT:
            # replace with web_docs
            docs = web_retriever.get_relevant_documents(question)

    # convert content to markdown
    context = ["* " + doc.page_content for doc in docs]
    context = "\n".join(context)

    # Phase 4: use updated context to generate answer
    prompt_template = read_template_from_file(CRAG_PROMPT_2_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })
    result = parse_response(response)
    qa_pair = result.value["qa_pair"]
    print("QAPair:", qa_pair)


if __name__ == "__main__":
    asyncio.run(pipeline())
