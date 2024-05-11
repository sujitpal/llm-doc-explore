import boto3
import os

from dotenv import find_dotenv, load_dotenv
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import BedrockChat
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever


DATA_DIR = "../data"
CHROMADB_PATH = os.path.join(DATA_DIR, "chroma-db")


_ = load_dotenv(find_dotenv())

boto3_bedrock = boto3.client('bedrock-runtime')
model_id = "anthropic.claude-v2"

llm = BedrockChat(
    model_id=model_id,
    model_kwargs={
        "temperature": 0.1
    })

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cpu"},
                                  encode_kwargs={"normalize_embeddings": True})
vectorstore = Chroma(persist_directory=CHROMADB_PATH,
                     embedding_function=embedding)

question = "The Snowflake free trial account allows you to use almost all the functionality of a paid account. What are a few of the differences, though, when using a Snowflake trial account?"


# Max Margin Relevance: address diversity

docs = vectorstore.max_marginal_relevance_search(question, fetch_k=10, k=3)

print("--- context 1 ---")
for i, doc in enumerate(docs):
    print("{:d}. {:s}".format(i, doc.page_content))

qdocs = "".join([doc.page_content for doc in docs])
response = llm.invoke(f"{qdocs} Question: {question}")
print("--- response: 1 ---")
print(response)


# # Self Query Retriever: extract filter from query using LLM
# # (probably needs more things to line up, gives parsing exception)

# field_info = [
#     AttributeInfo(
#         name="source",
#         description="the chapter file name the text comes from",
#         type="string"
#     ),
#     AttributeInfo(
#         name="chapter",
#         description="numeric chapter number in the book",
#         type="string"
#     )
# ]
# content_info = "chunks from book chapters from book about Snowflake DB"
# retriever = SelfQueryRetriever.from_llm(
#     llm=llm,
#     vectorstore=vectorstore,
#     document_contents=content_info,
#     metadata_field_info=field_info,
#     verbose=True
# )
# docs = retriever.get_relevant_documents(question)
# print("--- context 2 ---")
# for i, doc in enumerate(docs):
#     print("{:d}. {:s}".format(i, doc.page_content))

# qdocs = "".join([doc.page_content for doc in docs])
# response = llm.invoke(f"{qdocs} Question (answer from chapter 1): {question}")
# print("--- response: 2 ---")
# print(response)

# # Contextual Compression

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
docs = compression_retriever.get_relevant_documents(question)
print("--- context 3 ---")
for i, doc in enumerate(docs):
    print("{:d}. {:s}".format(i + 1, doc.page_content))

qdocs = "".join([doc.page_content for doc in docs])
response = llm.invoke(f"{qdocs} Question: {question}")
print("--- response: 3 ---")
print(response)


# Contextual Compression + MMR
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_type="mmr")
)
docs = compression_retriever.get_relevant_documents(question)
print("--- context 4 ---")
for i, doc in enumerate(docs):
    print("{:d}. {:s}".format(i + 1, doc.page_content))

qdocs = "".join([doc.page_content for doc in docs])
response = llm.invoke(f"{qdocs} Question: {question}")
print("--- response: 4 ---")
print(response)
