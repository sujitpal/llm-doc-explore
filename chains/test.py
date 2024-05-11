import langchain
import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from operator import itemgetter
from pydantic import BaseModel, Field

from chain_utils import (
    extract_questions, sample_question,
    read_template_from_file, parse_response
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
RAG_CHAIN_PROMPT_FP = "rag_chain.prompt.txt"


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                      model_kwargs={"device": "cpu"},
                                      encode_kwargs={
                                          "normalize_embeddings": True
                                      })
    vectorstore = Chroma(persist_directory=CHROMA_DIR,
                         embedding_function=embedding)

    model = BedrockChat(
        model_id="anthropic.claude-v2",
        model_kwargs={
            "temperature": 0.0
        })

    prompt_template = read_template_from_file(RAG_CHAIN_PROMPT_FP)

    langchain.debug = False

    questions = extract_questions(CHAPTERS_DIR)
    chapter, question = sample_question(questions)
    print(chapter, question)

    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_kwargs={
            "filter": dict(chapter=int(chapter)),
            "k": 5
        },
        search_type="mmr"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"],
    )

    chain = (
        {
            "context": itemgetter("question") | retriever,
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
    print(result)
    qapair = result.value["qa_pair"]
    print("QAPair:", qapair)
