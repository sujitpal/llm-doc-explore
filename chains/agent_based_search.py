import os
import requests

from dotenv import find_dotenv, load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, tool
from langchain.agents.output_parsers import XMLAgentOutputParser
from langchain_core.language_models.llms import LLM
from langchain_community.chat_models import BedrockChat
from typing import List

from my_retrievers import (
    create_base_vector_retriever,
    convert_query_to_keyphrases,
    WebSearchRetriever
)


DATA_DIR = "../data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma-db")
CK_SOLR_URL = "http://ck-search-cert-elb-global-public.clinicalkey.com:8080/solr/ck/select"

MODEL = None
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={
                "temperature": 0.0
            })
    return MODEL


@tool
def search_medical(query: str) -> List[str]:
    """ Return information about provided query that are medical in nature.
    """
    headers = {"Content-type": "application/json"}
    keyphrases = convert_query_to_keyphrases(query, get_model())
    query_k = " ".join(keyphrases)
    params = {
        "q": query_k,
        "wt": "json",
        "fl": "summary_s",
        "fq": [
            "lang:eng",
            "contenttype:(BK JL)",
            "product=physician"
        ]
    }
    response = requests.get(CK_SOLR_URL, headers=headers, params=params)
    docs = response.json()["response"]["docs"]
    results = [doc["summary_s"] for doc in docs]
    return results


@tool
def search_snowflake(query: str) -> List[str]:
    """ Return information about provided query about technical details
        about Snowflake Data Warehouse and how to use it.
    """
    vector_retriever = create_base_vector_retriever(CHROMA_DIR)
    vector_retriever.search_kwargs = {"k": 3}
    docs = vector_retriever.get_relevant_documents(query)
    results = [doc.page_content for doc in docs]
    return results


@tool
def search_general(query: str) -> List[str]:
    """ Return general information about provided query from the web.
    """
    retriever = WebSearchRetriever.create(get_model())
    docs = retriever.get_relevant_documents(query)
    results = [doc.page_content for doc in docs]
    return results


# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}"
                      for tool in tools])


if __name__ == "__main__":

    _ = load_dotenv(find_dotenv())

    tool_list = [
        search_snowflake,
        search_medical,
        search_general
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/xml-agent-convo")
    # print(prompt)
    # You are a helpful assistant. Help the user answer any questions.
    #
    # You have access to the following tools:
    #
    # {tools}
    #
    # In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> 
    # tags. You will then get back a response in the form <observation></observation>.
    #
    # For example, if you have a tool called 'search' that could run a google search,
    # in order to search for the weather in SF you would respond:
    #
    # <tool>search</tool><tool_input>weather in SF</tool_input>
    # <observation>64 degrees</observation>
    #
    # When you are done, respond with a final answer between <final_answer></final_answer>. 
    # For example:
    #
    # <final_answer>The weather in SF is 64 degrees</final_answer>
    #
    # Begin!
    #
    # Previous Conversation:
    #
    # {chat_history}
    #
    # Question: {input}
    # {agent_scratchpad}
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: convert_intermediate_steps(
                x["intermediate_steps"]
            ),
        }
        | prompt.partial(tools=convert_tools(tool_list))
        | get_model().bind(stop=["</tool_input>", "</final_answer>"])
        | XMLAgentOutputParser()
    )
    queries = [
        "What is unique about the SNOWFLAKE database that is included in the account?",
        "I accidentally ingested diarhea bacteria in melted snow flakes, what to do?",
        "What is the trend for Yahoo stock price over last week?"
    ]

    for query in queries:
        agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
        response = agent_executor.invoke({"input": query})
        print(response)
