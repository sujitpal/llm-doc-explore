import boto3
import langchain
import os

from dotenv import find_dotenv, load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pydantic import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain.output_parsers import XMLOutputParser
from typing import Sequence

_ = load_dotenv(find_dotenv())

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


class QAPair(BaseModel):
    question: str = Field(description="generated question")
    answer: str = Field(description="answer to question")


class QAPairs(BaseModel):
    qa_pairs: Sequence[QAPair] = Field(description="List of QAPair objects")

# pydantic_object = QAPairs


# # output_parser = PydanticOutputParser(pydantic_object=Joke)
# output_parser = PydanticOutputParser(pydantic_object=QAPairs)

# print("--- format instructions ---")
# print(output_parser.get_format_instructions())
# print("---")

# # xml_output_parser = XMLOutputParser(tags=["qa_pairs", "qa_pair", "question", "answer"])
# # print(xml_output_parser.get_format_instructions())

format_instructions = """
Format your Joke as well-formatted XML follows:
<Joke>
    <setup>question to set up as a joke</setup>
    <punchline>answer to resolve the joke</punchline>
</Joke>
"""

# prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\ntell me a short joke about {topic}",
    input_variables=["format_instructions", "topic"],
    # partial_variables={
    #     "format_instructions": output_parser.get_format_instructions()
    # },
)

# boto3_bedrock = boto3.client('bedrock-runtime')
model_id = "anthropic.claude-v2"
model = BedrockChat(
    model_id=model_id,
    model_kwargs={
        "temperature": 0.3
    }
)

# output_parser = StrOutputParser()
# output_parser = PydanticOutputParser(pydantic_object=QAPairs)
output_parser = XMLOutputParser(tags=["Joke", "setup", "punchline"])


chain = prompt | model | output_parser

result = chain.invoke({"topic": "ice cream",
                       "format_instructions": format_instructions})
print(result)

