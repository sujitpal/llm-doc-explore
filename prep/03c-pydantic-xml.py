import json

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import Sequence, Optional, Dict


def json2xml(json_obj, line_padding=""):
    result_list = list()
    json_obj_type = type(json_obj)
    if json_obj_type is list:
        for sub_elem in json_obj:
            result_list.append(json2xml(sub_elem, line_padding))
        return "\n".join(result_list)
    if json_obj_type is dict:
        for tag_name in json_obj:
            sub_obj = json_obj[tag_name]
            result_list.append("%s<%s>" % (line_padding, tag_name))
            result_list.append(json2xml(sub_obj, "\t" + line_padding))
            result_list.append("%s</%s>" % (line_padding, tag_name))
        return "\n".join(result_list)
    return "%s%s" % (line_padding, json_obj)


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


class QAPair(BaseModel):
    question: str = Field(description="generated question")
    answer: str = Field(description="answer to question")


class QAPairs(BaseModel):
    qa_pairs: Sequence[QAPair] = Field(description="List of QAPair objects")


# schema_dict = QAPairs.model_json_schema()
# schema_dict = QAPairs.schema()
schema_dict = Joke.schema()
print("---")
print(json.dumps(schema_dict, indent=2))

# schema_json = json.loads(schema_dict)
schema_xml = json2xml(schema_dict)

print("---")
print(schema_xml)
