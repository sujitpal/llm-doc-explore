import boto3
import json
import langchain
import os

from dotenv import find_dotenv, load_dotenv
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

from chain_utils import (
    read_template_from_file, parse_response,
    extract_chapter_from_name
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
SUMMARY_FP = os.path.join(DATA_DIR, "chapter_summaries.jsonl")
SUMMARIZE_PROMPT_1_FP = "prog_summary_1.prompt.txt"
SUMMARIZE_PROMPT_2_FP = "prog_summary_2.prompt.txt"


if True:

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

    fout = open(SUMMARY_FP, "w", encoding="utf-8")

    for chapter_fn in os.listdir(CHAPTERS_DIR):
        if not chapter_fn.startswith("chapter-"):
            continue
        chapter_fp = os.path.join(CHAPTERS_DIR, chapter_fn)
        chap_id = extract_chapter_from_name(chapter_fn, is_question=False)
        if chap_id is None or len(chap_id.strip()) == 0:
            continue
        text_loader = TextLoader(chapter_fp)
        chapter_docs = text_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=1000,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(chapter_docs)
        print("==== chapter {:s}, #-chunks: {:d} ====".format(
            chap_id, len(chunks)))
        prev_summary = None
        for cid, chunk in enumerate(chunks):
            if prev_summary is None:
                prompt_template = read_template_from_file(
                    SUMMARIZE_PROMPT_1_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["text"]
                )
                inputs = {"text": chunk.page_content}
            else:
                prompt_template = read_template_from_file(
                    SUMMARIZE_PROMPT_2_FP)
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["previous_summary", "new_text"]
                )
                inputs = {
                    "previous_summary": prev_summary,
                    "new_text": chunk.page_content
                }
            chain = prompt | model | StrOutputParser()
            response = chain.invoke(inputs)
            result = parse_response(response)
            summary = result.value["summary"]
            print("--- chapter {:s}, chunk {:d} output ---".format(
                chap_id, cid))
            print(summary)
            prev_summary = summary
        fout.write(json.dumps({
            "chapter": chap_id,
            "summary": summary
        }) + "\n")

    fout.close()
