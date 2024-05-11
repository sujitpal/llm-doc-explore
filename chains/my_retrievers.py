import asyncio

from enum import Enum
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import List, Union

from chain_utils import (
    parse_response, read_template_from_file
)


WEBSEARCH_PROMPT_FP = "websearch.prompt.txt"
HYDE_PROMPT_FP = "hyde.prompt.txt"
RAG_FUSION_PROMPT_FP = "rag_fusion.prompt.txt"


class RetrieverType(Enum):
    VECTOR = "vector"
    LEX_VEC_SEQ = "lex_vec_seq"
    LEX_VEC_PAR = "lex_vec_par"
    WEB = "web"
    HYDE = "hyde"
    RAG_FUSION = "rag_fusion"
    PARENT = "parent"


class QList(BaseModel):
    queries: List[str] = Field(alias="queries",
                               description="list of generated queries")


def create_base_vector_retriever(chroma_db_dir: str):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                      model_kwargs={"device": "cpu"},
                                      encode_kwargs={
                                          "normalize_embeddings": True
                                      })
    vectorstore = Chroma(persist_directory=chroma_db_dir,
                         embedding_function=embedding)
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore,
        search_kwargs={
            "k": 5
        },
        search_type="mmr"
    )
    return retriever


def convert_query_to_keyphrases(query: str,
                                model: Union[LLM, BaseChatModel]
                                ) -> List[str]:
    prompt_template = read_template_from_file(WEBSEARCH_PROMPT_FP)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({
        "question": query
    })
    keyphrases = []
    if response is not None:
        keyphrases = [x.strip() for x in response.strip().split(",")]
    return keyphrases


def merge_docs_using_rrf(
        docs_list: List[List[Document]]
        ) -> List[Document]:
    scored_docs = {}
    top_k = max([len(docs) for docs in docs_list])
    for docs in docs_list:
        for pos, doc in enumerate(docs):
            print(pos, doc)
            doc_id = doc.metadata["doc_id"]
            if doc_id in scored_docs.keys():
                prev_score = scored_docs[doc_id][0]
                score = prev_score + (1.0 / (pos + 1))
            else:
                score = 1.0 / (pos + 1)
            scored_docs[doc_id] = (score, doc)
    rrf_sorted_docs = sorted(scored_docs.values(),
                             key=itemgetter(0), reverse=True)
    merged_docs = []
    for score, doc in rrf_sorted_docs[0:top_k]:
        doc.metadata["score"] = score
        merged_docs.append(doc)
    return merged_docs


class LexicalVectorSequenceRetriever(BaseRetriever):
    tfidf_retriever: TFIDFRetriever = None
    vector_retriever: VectorStoreRetriever = None

    @classmethod
    def create(cls,
               _tfidf_retriever: TFIDFRetriever,
               _vector_retriever: VectorStoreRetriever
               ):
        tfidf_retriever = _tfidf_retriever
        vector_retriever = _vector_retriever
        return cls(tfidf_retriever=tfidf_retriever,
                   vector_retriever=vector_retriever)
        
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
        # figure out the chapter using a BM25 search on chapters
        chapter = None
        docs = self.tfidf_retriever.get_relevant_documents(query)
        for doc in docs:
            chapter = doc.metadata["chapter"]
            break
        # retrieve context from vector store
        search_kwargs = {"k": 5}
        if chapter is not None:
            search_kwargs["filter"] = dict(chapter=chapter)
        print("setting search_kwargs:", search_kwargs)
        self.vector_retriever.search_kwargs = search_kwargs
        docs = self.vector_retriever.get_relevant_documents(query)
        return docs


class LexicalVectorParallelRRFRetriever(BaseRetriever):
    tfidf_retriever: TFIDFRetriever = None
    vector_retriever: VectorStoreRetriever = None

    @classmethod
    def create(cls,
               _tfidf_retriever: TFIDFRetriever,
               _vector_retriever: VectorStoreRetriever):
        tfidf_retriever = _tfidf_retriever
        vector_retriever = _vector_retriever
        return cls(tfidf_retriever=tfidf_retriever,
                   vector_retriever=vector_retriever)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
        
        lexical_docs = self.tfidf_retriever.get_relevant_documents(query)
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        docs_list = [lexical_docs, vector_docs]
        merged_docs = merge_docs_using_rrf(docs_list)
        return merged_docs


class WebSearchRetriever(BaseRetriever):
    model: Union[LLM, BaseChatModel] = None

    @classmethod
    def create(cls, _model: LLM):
        model = _model
        return cls(model=model)

    def _get_relevant_documents(self,
                                query: str, *, 
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> List[Document]:
        key_phrases = convert_query_to_keyphrases(query, self.model)
        search = SearchApiAPIWrapper()
        keyphrases_quoted = ["\"{:s}\"".format(kp) for kp in key_phrases]
        query_str = " ".join(keyphrases_quoted)
        print("running query:", query_str)
        response = search.run(query_str)
        if isinstance(response, list):
            docs = [Document(page_content=r) for r in response]
        else:
            docs = [Document(page_content=response)]
        return docs


class HydeRetriever(BaseRetriever):
    """ based on paper: https://arxiv.org/abs/2212.10496 """
    model: LLM = None
    vector_retriever: VectorStoreRetriever = None

    @classmethod
    def create(cls,
               _model: LLM, 
               _vector_retriever: VectorStoreRetriever):
        model = _model
        vector_retriever = _vector_retriever
        return cls(model=model, vector_retriever=vector_retriever)

    def _get_relevant_documents(self,
                                query: str, *, 
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> List[Document]:
        prompt_template = read_template_from_file(HYDE_PROMPT_FP)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"]
        )
        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({
            "question": query
        })
        result = parse_response(response)
        hyde_query = result.value["passage"]
        print("HyDE query:", hyde_query)
        docs = self.vector_retriever.get_relevant_documents(hyde_query)
        return docs


class RAGFusionRetriever(BaseRetriever):
    """ based on github: https://github.com/Raudaschl/rag-fusion
        and paper: https://arxiv.org/abs/2402.03367 
    """
    model: LLM = None
    tfidf_retriever: TFIDFRetriever = None
    vector_retriever: VectorStoreRetriever = None

    @classmethod
    def create(cls,
               _model: LLM,
               _tfidf_retriever: TFIDFRetriever,
               _vector_retriever: VectorStoreRetriever):
        model = _model
        tfidf_retriever = _tfidf_retriever
        vector_retriever = _vector_retriever
        return cls(model=model,
                   tfidf_retriever=tfidf_retriever,
                   vector_retriever=vector_retriever)
    
    async def _get_relevant_documents_for_query_variation(
            self, query: str,
            retriever: LexicalVectorSequenceRetriever
            ) -> List[Document]:
        docs = retriever.get_relevant_documents(query)
        return docs

    async def _run_query_variations_in_parallel(self, queries, retriever):
        tasks = []
        for qid, query in enumerate(queries):
            print("query variation {:d}: {:s}".format(qid, query))
            tasks.append(asyncio.create_task(
                self._get_relevant_documents_for_query_variation(
                    query, retriever)))
        docs_list = await asyncio.gather(*tasks)
        return docs_list

    def _get_relevant_documents(self,
                                query: str, *,
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> List[Document]:
        retriever = LexicalVectorSequenceRetriever.create(
            self.tfidf_retriever, self.vector_retriever)
        
        prompt_template = read_template_from_file(RAG_FUSION_PROMPT_FP)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"]
        )
        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({
            "question": query
        })
        result = parse_response(response)
        queries = result.value["queries"]["query"]
        loop = asyncio.get_event_loop()
        docs_list = loop.run_until_complete(
            self._run_query_variations_in_parallel(queries, retriever))
        docs = merge_docs_using_rrf(docs_list)
        return docs


class ParentDocumentRetriever(BaseRetriever):
    vector_retriever: VectorStoreRetriever = None
    parent_retriever: TFIDFRetriever = None

    @classmethod
    def create(cls,
               _vector_retriever: VectorStoreRetriever,
               _parent_retriever: TFIDFRetriever):
        vector_retriever = _vector_retriever
        parent_retriever = _parent_retriever
        return cls(vector_retriever=vector_retriever,
                   parent_retriever=parent_retriever)

    def _get_relevant_documents(self,
                                query: str, *,
                                run_manager: CallbackManagerForRetrieverRun
                                ) -> List[Document]:

        search_kwargs = {"k": 10}
        self.vector_retriever.search_kwargs = search_kwargs
        chunk_docs = self.vector_retriever.get_relevant_documents(query)
        
        chapters = set()
        for chunk in chunk_docs:
            chapter = chunk.metadata["chapter"]
            chapters.add(chapter)
        chapters = list(chapters)
        print("found chapters:", chapters)

        self.parent_retriever.k = 3
        self.parent_retriever.metadata = {
            "chapter": {
                "$in": chapters
            }
        }
        docs = self.parent_retriever.get_relevant_documents(query)
        return docs


# class FooRetriever(BaseRetriever):

#     def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
#         return super()._get_relevant_documents(query, run_manager=run_manager)    