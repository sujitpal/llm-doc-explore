import argparse
import boto3
import langchain
import os

from dotenv import find_dotenv, load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory
)
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import BaseModel, Field

from chain_utils import (
    read_template_from_file, parse_response
)
from my_retrievers import (
    create_base_vector_retriever,
    LexicalVectorSequenceRetriever
)


DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
TFIDF_CHAP_DIR = os.path.join(DATA_DIR, "tfidf-chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
RAG_PROMPT_FP = "conv_rag_chain_1.prompt.txt"
REQ_PROMPT_FP = "conv_rag_chain_2.prompt.txt"

CONV_QUESTIONS = [
    "Which Snowflake function can you use to see the clustering depth in a table?",
    "How does Snowflake determine how well clustered it is?",
    "What is a good rule of thumb when deciding on a clustering key for this?",
    "What is the maximum number of columns or expressions you should choose for such a key?",
]


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


class QQPair(BaseModel):
    original_question: str = Field(alias="original_question",
                                   description="input question")
    rephrased_question: str = Field(alias="rephrased_question",
                                    description="rephrased question")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=False,
                        choices=["buffer", "window", "token", "summary"],
                        default="buffer")
    args = parser.parse_args()

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
            "temperature": 0.3,
            "max_tokens_to_sample": 1024,
            "stop_sequences": ["\n\nHuman"]
        }
    )

    memory = None
    match args.memory:
        case "buffer":
            memory = ConversationBufferMemory(
                human_prefix="in", ai_prefix="out")
        case "window":
            memory = ConversationBufferWindowMemory(
                human_prefix="in", ai_prefix="out")
        case "token":
            memory = ConversationTokenBufferMemory(
                human_prefix="in", ai_prefix="out", llm=model)
        case "summary":
            memory = ConversationSummaryMemory(
                human_prefix="in", ai_prefix="out", llm=model)
            
    rag_prompt_template = read_template_from_file(RAG_PROMPT_FP)
    rag_prompt = PromptTemplate(
        template=rag_prompt_template,
        input_variables=["question", "context"])

    req_prompt_template = read_template_from_file(REQ_PROMPT_FP)
    req_prompt = PromptTemplate(
        template=req_prompt_template,
        input_variables=["chat_history", "question"]
    )

    langchain.debug = False

    for question in CONV_QUESTIONS:

        # rewrite query if there is chat history
        if len(memory.chat_memory.messages) > 0:
            req_chain = req_prompt | model | StrOutputParser()
            response = req_chain.invoke({
                "chat_history": memory.load_memory_variables({}),
                "question": question
            })
            print("--- rewriting query ---")
            # print(response)
            result = parse_response(response)
            qq_pair = result.value["qq_pair"]
            print("QQPair:", qq_pair)
            question = qq_pair["rephrased_question"]

        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | rag_prompt
            | model
            | StrOutputParser()
        )
        response = rag_chain.invoke({
            "question": question
        })
        print("--- reply to question ---")
        # print(response)
        result = parse_response(response)
        qa_pair = result.value["qa_pair"]
        print(qa_pair)

        memory.save_context(
            {"input": qa_pair["question"]},
            {"output": qa_pair["answer"]}
        )

