import asyncio
import boto3
import langchain
import numpy as np
import os

from dotenv import find_dotenv, load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import List, Tuple

from chain_utils import (
    read_template_from_file, parse_response
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
QGEN_PROMPT = "qgen_eval_chain_1.prompt.txt"
QANS_PROMPT = "qgen_eval_chain_2.prompt.txt"
QEVAL_PROMPT = "qgen_eval_chain_3.prompt.txt"


class QAPair(BaseModel):
    question: str = Field(alias="question", description="Question to ask")
    answer: str = Field(alias="answer", description="Answer text")


class QAPairs(BaseModel):
    qa_pairs: List[QAPair] = Field(alias="qa_pair",
                                   description="list of QAPair objects")


class QATriple(BaseModel):
    question: str = Field(alias="question", description="question text")
    generated_answer: str = Field(alias="generated_answer",
                                  description="answer generated by QG chain")
    predicted_answer: str = Field(alias="predicted_answer",
                                  description="answer predicted by QA chain")


class QAEval(BaseModel):
    question: str = Field(alias="question", description="question text")
    student_answer: str = Field(alias="student_answer",
                                description="answer predicted by QA chain")
    true_answer: str = Field(alias="true_answer",
                             description="answer generated by QG chain")
    explanation: str = Field(alias="explanation",
                             description="chain of thought for grading")
    grade: str = Field(alias="grade",
                       description="LLM grade CORRECT or INCORRECT")


async def predict_answers(context: str,
                          qa_pairs: List[QAPair],
                          chain: Runnable) -> List[QATriple]:
    tasks, generated_answers = [], []
    for qa_pair in qa_pairs:
        generated_answers.append(qa_pair["answer"])
        tasks.append(chain.ainvoke({
            "question": qa_pair["question"],
            "context": context,
        }))
    responses = await asyncio.gather(*tasks)
    qa_triples = []
    for response, generated_answer in zip(responses, generated_answers):
        result = parse_response(response)
        print("---")
        print(result)
        predicted_qa_pair = result.value["qa_pair"]
        qa_triples.append(QATriple(
            question=predicted_qa_pair["question"],
            generated_answer=generated_answer,
            predicted_answer=predicted_qa_pair["answer"]
        ).dict())
    return qa_triples


async def evaluate_questions(context: str,
                             qa_triples: List[QATriple],
                             chain: Runnable) -> Tuple[int, int]:
    tasks = []
    for qa_triple in qa_triples:
        tasks.append(chain.ainvoke({
            "question": qa_triple["question"],
            "context": context,
            "predicted_answer": qa_triple["predicted_answer"],
            "generated_answer": qa_triple["generated_answer"]
        }))
    responses = await asyncio.gather(*tasks)
    num_tested, num_correct = 0, 0
    for response in responses:
        result = parse_response(response)
        print("---")
        print(result)
        qa_eval = result.value["qa_eval"]
        if qa_eval["grade"] == "CORRECT":
            num_correct += 1
        num_tested += 1
    return num_correct, num_tested


async def pipeline():
    _ = load_dotenv(find_dotenv())

    boto3_bedrock = boto3.client("bedrock-runtime")
    model = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3_bedrock,
        model_kwargs={
            "temperature": 0.3,
            "max_tokens_to_sample": 1024,
            "stop_sequences": ["\n\nHuman"]
        },
        streaming=True
    )

    langchain.debug = False

    print("--- Phase 1: generate questions from content ---")

    prompt_template = read_template_from_file(QGEN_PROMPT)

    chapter_fp = os.path.join(CHAPTERS_DIR, "chapter-10.txt")
    text_loader = TextLoader(chapter_fp)
    chapter_docs = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(chapter_docs)
    print(len(chunks))
    random_chunk = np.random.randint(0, len(chunks))
    context = chunks[random_chunk].page_content
    print(f"context: {context}")

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"],
    )

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "context": context
    })
    # print(response)
    result = parse_response(response)
    generated_qa_pairs = result.value["qa_pairs"]["qa_pair"]
    print(generated_qa_pairs)

    print("--- Phase 2: generate answers for generated questions ---")

    prompt_template = read_template_from_file(QANS_PROMPT)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = prompt | model | StrOutputParser()

    qa_triples = await predict_answers(context, generated_qa_pairs, chain)
    print(qa_triples)

    print("--- Phase 3: evaluate generated answers against original answers ---")

    prompt_template = read_template_from_file(QEVAL_PROMPT)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "question", "context", "predicted_answer", "generated_answer"
        ])
    chain = prompt | model | StrOutputParser()

    num_correct, num_tested = await evaluate_questions(
        context, qa_triples, chain)
    print("Evaluation results: {:d} / {:d} CORRECT, {:.3f}%".format(
        num_correct, num_tested, 100.0 * num_correct / num_tested))


if __name__ == "__main__":
    asyncio.run(pipeline())