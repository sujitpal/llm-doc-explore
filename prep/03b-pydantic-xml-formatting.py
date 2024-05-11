from pydantic_xml import BaseXmlModel
from langchain.pydantic_v1 import BaseModel, Field


from typing import List


# class Request(BaseXmlModel, tag='Request'):
#     raw_cookies: str = Field(exclude=True)
#     raw_auth: str = Field(exclude=True)


# request = Request(
#     raw_cookies="PHPSESSID=298zf09hf012fh2; csrftoken=u32t4o3tb3gg43;",
#     raw_auth="Basic YWxhZGRpbjpvcGVuc2VzYW1l",
# )
# print(request.to_xml())

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# print(Joke.__fields__)

# print(f"<{Joke.__name__}>")
# for field_name, field in Joke.__fields__.items():
#     print(f"<{field_name}>{field.field_info.description}</{field_name}>")
# print(f"</{Joke.__name__}>")

joke = Joke(
    setup="tell me a joke about ice-cream",
    punchline="we all scream for ice-cream"
)
print(joke.model_dump())

# class Joke(BaseXmlModel):
#     setup: str
#     punchline: str

# joke_obj = Joke(setup="question to set up a joke",
#                 punchline="answer to resolve the joke")
# print(joke_obj.to_xml())


# joke_obj = Joke.from_xml("""<Joke>
#     <setup>questions to set up the joke</setup>
#     <punchline>answer to resolve the joke</punchline>
# </Joke>""")
# print(joke_obj)
# xml = joke_obj.to_xml(
#     pretty_print=True,
#     encoding='UTF-8',
#     standalone=True
# )
# print(xml)

# # print(joke.to_xml())


