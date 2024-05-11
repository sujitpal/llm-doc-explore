import boto3
import langchain
import os

from dotenv import find_dotenv, load_dotenv
from langchain.evaluation.qa import QAGenerateChain
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain.evaluation.qa import QAEvalChain

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import BedrockChat
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Sequence


_ = load_dotenv(find_dotenv())

DATA_DIR = "../data"
CHROMADB_PATH = os.path.join(DATA_DIR, "chroma-db")

boto3_bedrock = boto3.client('bedrock-runtime')
model_id = "anthropic.claude-v2"

langchain.debug = True

llm = BedrockChat(
    model_id=model_id,
    model_kwargs={
        "temperature": 0.0
    })

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cpu"},
                                  encode_kwargs={"normalize_embeddings": True})
vectorstore = Chroma(persist_directory=CHROMADB_PATH,
                     embedding_function=embedding)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>"
    })


class QAPair(BaseModel):
    QUESTION: str = Field(description="generated question")
    ANSWER: str = Field(description="answer to question")


class QAPairs(BaseModel):
    qa_pairs: Sequence[QAPair] = Field(description="List of QAPair objects")


llm_qg = BedrockChat(
    model_id=model_id,
    model_kwargs={
        "temperature": 0.3
    })
output_parser = PydanticOutputParser(pydantic_object=QAPairs)
qg = QAGenerateChain.from_llm(llm_qg)    #, output_parser=output_parser)
# qg = QAGenerateChain.from_llm(llm_qg, output_parser=output_parser)

question = "The Snowflake free trial account allows you to use almost all the functionality of a paid account. What are a few of the differences, though, when using a Snowflake trial account?"
docs = vectorstore.max_marginal_relevance_search(question, fetch_k=10, k=3)

docs_qg = qg.batch([{"doc": doc} for doc in docs])
# gen_questions = qg.apply_and_parse(
#     [{"doc": t} for t in docs]
# )
print("--- generated questions ---")
print(docs_qg)

# predictions = qa.batch([{"query": doc["qa_pairs"]["query"]} 
#                         for doc in docs_qg])
# print("--- predictions ---")
# print(predictions)


# llm_eval = BedrockChat(
#     model_id=model_id,
#     model_kwargs={
#         "temperature": 0.0
#     })
# eval_chain = QAEvalChain.from_llm(llm_eval)

# examples = [doc["qa_pairs"] for doc in docs_qg]
# graded_outputs = eval_chain.evaluate(examples, predictions)
# print("--- graded outputs ---")
# print(graded_outputs)

# for i, graded_output in enumerate(graded_outputs):
#     print(f"--- Example {i} ---")
#     print(graded_output["results"])
