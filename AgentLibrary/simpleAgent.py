from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework import ChatAgent
from agent_framework import AgentThread

import os
from dotenv import load_dotenv

class SimpleAgent():
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME")
        self.apikey = os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        self.apiversion = os.getenv("AZURE_OPENAI_API_VERSION")

    def create_client(self) -> AzureOpenAIResponsesClient:
        client = AzureOpenAIResponsesClient(
            endpoint=self.endpoint,
            deployment_name=self.deployment,
            api_key=self.apikey,
            api_version=self.apiversion,
        )
        return client
    
    def create_agent(self, client:AzureOpenAIResponsesClient) -> ChatAgent:
        agent = client.create_agent(id="agent-01",
            name="simple agent",
            description="Simple Agent",
            instructions="You are a helpful agent. Answer the questions briefly",
            temperature=0.7,
            tools=[]
        )
        return agent
    
    def create_thread(self, agent: ChatAgent) -> AgentThread:
        thread = agent.get_new_thread()
        return thread
    
    async def run_agent(self, agent:ChatAgent, messages, thread=None):
        output = await agent.run(messages=messages, thread=thread)
        return output

