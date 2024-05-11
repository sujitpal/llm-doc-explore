import langchain
import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import TFIDFRetriever

from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import BaseModel, Field

from chain_utils import (
    extract_questions, sample_question,
    read_template_from_file, parse_response
)
from my_retrievers import (
    create_base_vector_retriever,
    LexicalVectorSequenceRetriever
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
TFIDF_CHAP_DIR = os.path.join(DATA_DIR, "tfidf-chapters")
RAG_CHAIN_PROMPT_FP = "rag_chain.prompt.txt"


class QAPair(BaseModel):
    question: str = Field(alias="question", description="question to ask")
    answer: str = Field(alias="answer", description="answer provided by LLM")


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    # instantiate combined retriever
    tfidf_retriever = TFIDFRetriever.load_local(TFIDF_CHAP_DIR)
    vector_retriever = create_base_vector_retriever(CHROMA_DIR)
    combined_retriever = LexicalVectorSequenceRetriever.create(
        tfidf_retriever, vector_retriever)

    model = BedrockChat(
        model_id="anthropic.claude-v2",
        model_kwargs={
            "temperature": 0.0
        })

    # generate a random question from set
    questions = extract_questions(CHAPTERS_DIR)
    _, question = sample_question(questions)
    print("question:", question)

    langchain.debug = False

    # use question and context to generate answer
    prompt_template = read_template_from_file(RAG_CHAIN_PROMPT_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"],
    )

    chain = (
        {
            "context": itemgetter("question") | combined_retriever,
            "question": itemgetter("question")
        }
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": question
    })
    print(response)
    result = parse_response(response)
    qapair = result.value["qa_pair"]
    print("QAPair:", qapair)
