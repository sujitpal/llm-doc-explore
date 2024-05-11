import boto3
import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Currently, bedrock-runtime is only available in a few regions.

_ = load_dotenv(find_dotenv())

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

boto3_bedrock = boto3.client('bedrock-runtime')
model_id = "anthropic.claude-v2"

chat = BedrockChat(
    model_id=model_id,
    model_kwargs={
        "temperature": 0.1
    })

# ---
# messages = [
#     SystemMessage(
#         content="""The following is a friendly conversation between a human and an AI.
#         The AI is talkative and provides lots of specific details from its context.
#         If the AI does not know the answer to a question, it truthfully says it 
#         does not know."""
#     ),
#     HumanMessage(
#         content="Hi there! I am a data scientist looking for a good conversation around unsupervised learning."
#     )
# ]
# messages = chat(messages)
# print(messages)
# print(type(messages))


# ---
# customer_review = """\
# This leaf blower is pretty amazing.  It has four settings:\
# candle blower, gentle breeze, windy city, and tornado. \
# It arrived in two days, just in time for my wife's \
# anniversary present. \
# I think my wife liked it so much she was speechless. \
# So far I've been the only one using it, and I've been \
# using it every other morning to clear the leaves on our lawn. \
# It's slightly more expensive than the other leaf blowers \
# out there, but I think it's worth it for the extra features.
# """

# review_template = """\
# For the following text, extract the following information:

# gift: Was the item purchased as a gift for someone else? \
# Answer True if yes, False if not or unknown.

# delivery_days: How many days did it take for the product \
# to arrive? If this information is not found, output -1.

# price_value: Extract any sentences about the value or price,\
# and output them as a comma separated Python list.

# Format the output as JSON with the following keys:
# gift
# delivery_days
# price_value

# text: {text}
# """

# prompt_template = ChatPromptTemplate.from_template(review_template)
# print(prompt_template)

# messages = prompt_template.format_messages(text=customer_review)
# response = chat(messages)
# print(response.content)


# gift_schema = ResponseSchema(name="gift",
#                              description="Was the item purchased\
#                              as a gift for someone else? \
#                              Answer True if yes,\
#                              False if not or unknown.")
# delivery_days_schema = ResponseSchema(name="delivery_days",
#                                       description="How many days\
#                                       did it take for the product\
#                                       to arrive? If this \
#                                       information is not found,\
#                                       output -1.")
# price_value_schema = ResponseSchema(name="price_value",
#                                     description="Extract any\
#                                     sentences about the value or \
#                                     price, and output them as a \
#                                     comma separated Python list.")

# response_schemas = [gift_schema, 
#                     delivery_days_schema,
#                     price_value_schema]

# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# output_dict = output_parser.parse(response.content)
# print("output_dict:", output_dict)


claude = Bedrock(
    model_id=model_id,
    client=boto3_bedrock,
    model_kwargs={"max_tokens_to_sample": 1000},
)

# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=claude,
#     memory=memory,
#     verbose=True
# )

# conversation.predict(input="Hi, my name is Andrew")
# conversation.predict(input="What is 1+1?")
# conversation.predict(input="What is my name?")
# print(memory.buffer)
# print(memory.load_memory_variables({}))

# memory = ConversationBufferMemory()
# memory.save_context({"input": "Hi"},
#                     {"output": "What's up"})
# print(memory.buffer)

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=claude, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
print(memory.load_memory_variables({}))
