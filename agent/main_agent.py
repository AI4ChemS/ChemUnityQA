import os, json, yaml
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from agent.GraphTool.kg_agent import run_goal

CFG = yaml.safe_load(open("configs_tobias/settings.yaml"))

@tool("kg.agent")
def kg_agent_tool(payload: Dict[str, Any]) -> str:
    res = run_goal(payload.get("goal",""), payload.get("constraints", {}))
    return json.dumps(res, ensure_ascii=False)

@tool("user_query_explainer")
def user_query_explainer_tool(payload: Dict[str, Any]) -> str:
    llm = ChatOpenAI(model=CFG["models"].get("chat","gpt-4o-mini"), temperature=0.1)
    q = payload.get("query","")
    prompt = ("You are a scientific consultant…\n" f"User Query: {q}\n")
    return llm.invoke(prompt).content

ORCH = ChatOpenAI(model=CFG["models"].get("chat","gpt-4o-mini"), temperature=0)
orchestrator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the orchestrator. For MOFs/CO2 selectivity/flue gas or literature-backed questions, "
     "CALL `kg.agent` with a clear goal and inferred constraints (e.g., temperature_K, humidity_RH_percent, mixture). "
     "Summarize tool results plainly for a materials scientist."),
    ("user", "{user_input}")
])

def create_my_agent():
    return orchestrator_prompt | ORCH.bind_tools([kg_agent_tool, user_query_explainer_tool]) | StrOutputParser()
