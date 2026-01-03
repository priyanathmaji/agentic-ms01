from AgentLibrary import simpleAgent
import asyncio

simple_agent = simpleAgent.SimpleAgent()
client = simple_agent.create_client()
agent = simple_agent.create_agent(client)
thread = simple_agent.create_thread(agent=agent)

output = asyncio.run(simple_agent.run_agent(agent,"My name is John and my age is 45", thread=thread))
print(output)
output = asyncio.run(simple_agent.run_agent(agent,"What is my name?", thread=thread))
print(output)
output = asyncio.run(simple_agent.run_agent(agent,"Whats my age", thread=thread))
print(output)
