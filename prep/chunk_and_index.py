import json
import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import TFIDFRetriever

from prep_utils import extract_chapter_from_fn

DATA_DIR = "../data"
INPUT_DIR = os.path.join(DATA_DIR, "chapters")

CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")

CHROMADB_PATH = os.path.join(DATA_DIR, "chroma-db")
TFIDF_CHUNK_INDEX_PATH = os.path.join(DATA_DIR, "tfidf-chunks")
TFIDF_CHAPTER_INDEX_PATH = os.path.join(DATA_DIR, "tfidf-chapters")


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                  model_kwargs={"device": "cpu"},
                                  encode_kwargs={"normalize_embeddings": True})

## create the index if it doesn't exist
all_chunk_docs, all_chapter_docs = [], []
if not os.path.exists(CHROMADB_PATH):

    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    os.makedirs(CHUNKS_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    for chapter_fn in os.listdir(INPUT_DIR):
        if not chapter_fn.startswith("chapter-"):
            continue

        # get the input text
        loader = TextLoader(os.path.join(INPUT_DIR, chapter_fn))
        chapter_docs = loader.load()
        assert len(chapter_docs) == 1

        all_chapter_docs.extend(chapter_docs)

        # remove < and > from page content to prevent issues later
        page_content = chapter_docs[0].page_content
        page_content = page_content.replace("<", "&lt;")
        page_content = page_content.replace(">", "&gt;")
        chapter_docs[0].page_content = page_content

        # set the chapter metadata
        chapter_id = extract_chapter_from_fn(chapter_fn)
        chapter_docs[0].metadata["chapter"] = int(chapter_id)

        # chunk the input text
        chunk_docs = text_splitter.split_documents(chapter_docs)
        print("chapter {:s}, #-chunks: {:d}".format(chapter_id, len(chunk_docs)))
        for cid, chunk_doc in enumerate(chunk_docs):
            chunk_doc.metadata["chunk"] = cid
            chunk_doc.metadata["doc_id"] = "{:s}-{:d}".format(chapter_id, cid)
            # write the Document objects as JSON
            chunk_fp = os.path.join(
                CHUNKS_DIR, "chunk-{:s}-{:d}.json".format(chapter_id, cid))
            with open(chunk_fp, "w", encoding="utf-8") as fch:
                fch.write(json.dumps(chunk_doc.to_json()) + "\n")

        # upload chunk documents into ChromaDB
        vector_db = Chroma.from_documents(documents=chunk_docs,
                                          embedding=embedding,
                                          persist_directory=CHROMADB_PATH)        
        vector_db.persist()
        all_chunk_docs.extend(chunk_docs)

    # tests to verify data loaded into vector store
    db = Chroma(persist_directory=CHROMADB_PATH, embedding_function=embedding)
    print("#-records uploaded to Chroma vector store:", db._collection.count())
    # #-records uploaded: 1701

    # TF-IDF index of chunks
    shutil.rmtree(TFIDF_CHUNK_INDEX_PATH, ignore_errors=True)
    os.makedirs(TFIDF_CHUNK_INDEX_PATH)
    tfidf_chunk_retriever = TFIDFRetriever.from_documents(all_chunk_docs)
    tfidf_chunk_retriever.save_local(TFIDF_CHUNK_INDEX_PATH)
    print("#-chunks loaded to chunk TF-IDF index:",
          len(tfidf_chunk_retriever.docs))

    # TF-IDF index of chapters
    shutil.rmtree(TFIDF_CHAPTER_INDEX_PATH, ignore_errors=True)
    os.makedirs(TFIDF_CHAPTER_INDEX_PATH)
    tfidf_chapter_retriever = TFIDFRetriever.from_documents(all_chapter_docs)
    tfidf_chapter_retriever.save_local(TFIDF_CHAPTER_INDEX_PATH)
    print("#-chunks loaded to chapter TF-IDF index:",
          len(tfidf_chapter_retriever.docs))


# do a test query to verify data loaded
question = "The Snowflake free trial account allows you to use almost all the functionality of a paid account. What are a few of the differences, though, when using a Snowflake trial account?"

filt = {"chapter" : 1}
results = db.similarity_search_with_score(question, filter=filt)
for doc in results:
    print(doc)
# (Document(page_content='instance. If you need information on how to create a 
#           free trial Snowflake account, refer to Appendix C .', 
#           metadata={'chapter': '1', 
#                     'source': '../data/chapters/chapter-1.txt'}
#           ), 0.47605377435684204)
# (Document(...)
