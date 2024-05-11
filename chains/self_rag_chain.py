# Self-RAG -- https://arxiv.org/abs/2310.11511
# blog post -- https://blog.langchain.dev/agentic-rag-with-langgraph/
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
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field
from typing import List

from chain_utils import (
    extract_questions, sample_question,
    parse_response, read_template_from_file
)
from my_retrievers import WebSearchRetriever


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
SELF_RAG_PROMPT_1_FP = "self_rag_chain_1.prompt.txt"
SELF_RAG_PROMPT_2_FP = "self_rag_chain_2.prompt.txt"
SELF_RAG_PROMPT_3_FP = "self_rag_chain_3.prompt.txt"
SELF_RAG_PROMPT_4_FP = "self_rag_chain_4.prompt.txt"
SELF_RAG_PROMPT_5_FP = "self_rag_chain_5.prompt.txt"

MAX_LOOPS = 3


class QCEval(BaseModel):
    question: str = Field(alias="question",
                          description="text of provided question")
    context: str = Field(alias="context", description="text of provided context")
    grade: str = Field(alias="grade", description="RELEVANT or NOT_RELEVANT")
    explanation: str = Field(alias="explanation",
                             description="reasoning for relevance judgment")


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


class QAEval(BaseModel):
    question: str = Field(alias="question",
                          description="text of provided question")
    answer: str = Field(alias="answer", description="text of provided answer")
    grade: str = Field(alias="grade", description="PASS or FAIL")
    explanation: str = Field(alias="explanation",
                             description="reasoning for judgment")


class AnsEval(BaseModel):
    answer: str = Field(alias="answer", description="answer text to evaluate")
    grade: str = Field(alias="grade", description="PASS or FAIL")
    explanation: str = Field(alias="explanation",
                             description="reasoning for judgment")


class QQPair(BaseModel):
    question: str = Field(alias="question", description="original question")
    rewritten: str = Field(alias="rewritten", description="rewritten question")


class States(Enum):
    ENTRY = -1
    RETRIEVE_CONTEXT = 0
    CHECK_CONTEXT_RELEVANCE = 1
    GENERATE_ANSWER = 2
    CHECK_SUPPORT = 3
    CHECK_USEFULNESS = 4
    REPHRASE_QUESTION = 5
    DO_WEB_SEARCH = 6
    EXIT = 99


async def evaluate_contexts(question: str,
                            docs: List[Document],
                            chain: Runnable) -> List[QCEval]:
    tasks = []
    for doc in docs:
        tasks.append(chain.ainvoke({
            "question": question,
            "context": doc.page_content
        }))
    responses = await asyncio.gather(*tasks)
    qc_evals = []
    for response in responses:
        result = parse_response(response)
        print("---")
        print(result)
        qc_eval = result.value["qc_eval"]
        qc_evals.append(QCEval(
            question=question,
            context=doc.page_content,
            grade=qc_eval["grade"],
            explanation=qc_eval["explanation"]
        ).dict())
    return qc_evals


def generate_answer(question: str,
                    qc_evals: List[QCEval],
                    chain: Runnable) -> str:
    context_list = []
    for qc_eval in qc_evals:
        context_list.append("* " + qc_eval["context"])
    context = "\n".join(context_list)
    response = chain.invoke({
        "question": question,
        "context": context
    })
    result = parse_response(response)
    print("result:", result)
    qa_pair = result.value["qa_pair"]
    answer = qa_pair["answer"]
    return answer


def check_if_supported(question: str,
                       answer: str,
                       chain: Runnable) -> bool:
    response = chain.invoke({
        "question": question,
        "answer": answer
    })
    result = parse_response(response)
    print("result:", result)
    qa_eval = result.value["qa_eval"]
    is_supported = qa_eval["grade"] == "PASS"
    return is_supported


def check_if_useful(answer: str,
                    chain: Runnable) -> bool:
    response = chain.invoke({
        "answer": answer
    })
    result = parse_response(response)
    print("result:", result)
    ans_eval = result.value["ans_eval"]
    is_useful = ans_eval["grade"] == "PASS"
    return is_useful


def rephrase_question(question: str,
                      chain: Runnable) -> str:
    response = chain.invoke({
        "question": question
    })
    result = parse_response(response)
    qq_pair = result.value["qq_pair"]
    return qq_pair["rewritten"]


def do_web_search(question: str,
                  chain: Runnable) -> List[str]:
    response = chain.invoke({
        "question": question
    })
    key_phrases = []
    if response is not None:
        key_phrases = [x.strip() for x in response.strip().split(",")]
    print("key phrases:", key_phrases)
    search = SearchApiAPIWrapper()
    response = search.run(" ".join(key_phrases))
    if isinstance(response, list):
        response = "\n".join(["* " + x for x in response])
    else:
        response = "* " + response
    return response


async def pipeline():

    _ = load_dotenv(find_dotenv())

    # set up vector only retriever
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                      model_kwargs={"device": "cpu"},
                                      encode_kwargs={
                                          "normalize_embeddings": True
                                      })
    vectorstore = Chroma(persist_directory=CHROMA_DIR,
                         embedding_function=embedding)
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_kwargs={
            "k": 10
        },
        search_type="mmr"
    )
    
    # LLM
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

    # select a question from the hopper
    questions = extract_questions(CHAPTERS_DIR)
    _, question = sample_question(questions)
    print("question:", question)

    curr_state = States.ENTRY
    num_loops = 0
    while True:
        if num_loops > MAX_LOOPS:
            curr_state = States.DO_WEB_SEARCH
        match curr_state:
            case States.ENTRY:
                print("--- Entering state machine ---")
                assert question is not None
                curr_state = States.RETRIEVE_CONTEXT
                continue
            case States.RETRIEVE_CONTEXT:
                print("--- Retrieving contents for question ---")
                docs = retriever.get_relevant_documents(question)
                print(docs)
                curr_state = States.CHECK_CONTEXT_RELEVANCE
                continue
            case States.CHECK_CONTEXT_RELEVANCE:
                print("--- Checking context relevance ---")
                prompt_template = read_template_from_file(SELF_RAG_PROMPT_1_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["question", "context"]
                )
                chain = prompt | model | StrOutputParser()
                qc_evals = await evaluate_contexts(question, docs, chain)
                print(qc_evals)
                relevant_qc_evals = [qc_eval for qc_eval in qc_evals
                                     if qc_eval["grade"] == "RELEVANT"]
                print("#-relevant context:", len(relevant_qc_evals))
                if len(relevant_qc_evals) == 0:
                    curr_state = States.DO_WEB_SEARCH
                else:
                    curr_state = States.GENERATE_ANSWER
                continue
            case States.GENERATE_ANSWER:
                print("--- Generating Answer ---")
                prompt_template = read_template_from_file(SELF_RAG_PROMPT_2_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["question", "context"]
                )
                chain = prompt | model | StrOutputParser()
                answer = generate_answer(question, relevant_qc_evals, chain)
                curr_state = States.CHECK_SUPPORT
                continue
            case States.CHECK_SUPPORT:
                print("--- Checking support for answer ---")
                prompt_template = read_template_from_file(SELF_RAG_PROMPT_3_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["question", "answer"]
                )
                chain = prompt | model | StrOutputParser()
                is_supported = check_if_supported(question, answer, chain)
                print("is_supported:", is_supported)
                if is_supported:
                    curr_state = States.CHECK_USEFULNESS
                else:
                    curr_state = States.REPHRASE_QUESTION
                continue
            case States.CHECK_USEFULNESS:
                print("--- Checking usefulness of answer ---")
                prompt_template = read_template_from_file(SELF_RAG_PROMPT_4_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["answer"]
                )
                chain = prompt | model | StrOutputParser()
                is_useful = check_if_useful(answer, chain)
                print("is_useful:", is_useful)
                if is_useful:
                    curr_state = States.EXIT
                else:
                    curr_state = States.REPHRASE_QUESTION
                continue
            case States.REPHRASE_QUESTION:
                print("--- Rephrasing question ---")
                prompt_template = read_template_from_file(SELF_RAG_PROMPT_5_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["question"]
                )
                chain = prompt | model | StrOutputParser()
                question_r = rephrase_question(question, chain)
                print("rephrased question:", question_r)
                question = question_r
                curr_state = States.RETRIEVE_CONTEXT
                num_loops += 1
                continue
            case States.DO_WEB_SEARCH:
                print("--- Doing web search ---")
                docs = web_retriever.get_relevant_documents(question)
                answer = docs[0].page_content
                curr_state = States.EXIT
            case States.EXIT:
                break

    print("answer:", answer)


if __name__ == "__main__":
    asyncio.run(pipeline())
