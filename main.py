import argparse
import os

from langchain_core.messages import convert_to_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

def create_my_agent(tools, prompt: str = None, use_memory: bool = False):
    # Create the agent with the tool
    mem = InMemorySaver() if use_memory else None

    agent = create_react_agent(
        model="gpt-4o",
        tools=tools,
        prompt= prompt if prompt else "You are helpful assistant",
        checkpointer = mem,
    )

    return agent

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", action="store_true",
                        help="triggers the agent in chat mode. Type quit to the agent to exit")
    parser.add_argument("--prompt", help="The question for the agent using string quotes")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pass