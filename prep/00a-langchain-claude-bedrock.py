import boto3
import os

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

# Currently, bedrock-runtime is only available in a few regions.

_ = load_dotenv(find_dotenv())

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

boto3_bedrock = boto3.client('bedrock-runtime')
modelId = "anthropic.claude-v2"

claude = Bedrock(
    model_id=modelId,
    client=boto3_bedrock,
    model_kwargs={"max_tokens_to_sample": 1000},
)
memory = ConversationBufferMemory()

# Turn verbose to true to see the full logs and documents
conversation = ConversationChain(
    llm=claude,
    verbose=False,
    memory=ConversationBufferMemory()  # memory_chain
)

# Langchain prompts do not always work with all the models. This prompt is tuned for Claude
claude_prompt = PromptTemplate.from_template("""

Human: The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. If the AI does not know
the answer to a question, it truthfully says it does not know.

Current conversation:
<conversation_history>
{history}
</conversation_history>

Here is the human's next reply:
<human_reply>
{input}
</human_reply>

Assistant:
""")

conversation.prompt = claude_prompt

human_reply = "Hi there! I am a data scientist looking for a good conversation around unsupervised learning."
history = ""
while human_reply != "quit":
    result = conversation.predict(input=human_reply, history=history)
    print(result)
    print("history:", history)
    human_reply = input()
