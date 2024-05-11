import boto3
import os
import langchain

from dotenv import find_dotenv, load_dotenv
from langchain.chains import RetrievalQA
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import BedrockChat
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

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


# # ---- approach 1 ----

# docs = vectorstore.similarity_search(question, filter=dict(chapter="1"))
# qdocs = "".join([doc.page_content for doc in docs])

# response = llm.invoke(f"{qdocs} Question: {question}")
# print("--- response: 1 ---")
# print(response)

# # ---- approach 2a ----

# retriever = vectorstore.as_retriever()    # no filtering
# qa_stuff = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     verbose=True
# )
# response = qa_stuff.invoke(question)
# print("--- response: 2a ---")
# print(response)


# ---- approach 2b ----
class ChromeFilteredRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> List[Document]:
        return vectorstore.similarity_search(query, k=5,
                                             filter=dict(chapter="1"))


langchain.debug = True
retriever = ChromeFilteredRetriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
response = qa_stuff.invoke(question)
print("--- response: 2b ---")
print(response)


# # ---- approach 3a ----
# index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# response = index.query(question, llm=llm)  # no filtering
# print("--- response 3a ---")
# print(response)

# # ---- approach 3b ----
# # with filtering
# retriever_kwargs = {
#     "search_kwargs": {
#         "filter": {
#             "chapter": "1"
#         }
#     }
# }
# response = index.query(question, llm=llm,
#                        retriever_kwargs=retriever_kwargs)
# print("--- response 3b ---")
# print(response)
