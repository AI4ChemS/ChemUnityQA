from typing import Dict, Any, List
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate # Ensure this import is here

# Import your custom tool
from agent.GraphTool.GraphRetriever import query_mof_database


# --- TOOL DEFINITION ---
# This part remains the same.
@tool
def mof_database_tool(question: str) -> str:
    """
    Query the MOF research database for information about Metal-Organic Frameworks,
    their properties, applications, and structure-property relationships.
    Use this for any questions about MOFs, selectivity, synthesis, or performance.
    """
    return query_mof_database(question)

tools = [mof_database_tool]


# --- AGENT "BRAIN" / PROMPT ---
# This is the new section that instructs the agent to plan and create subqueries.
AGENT_SYSTEM_PROMPT = """
You are an expert materials scientist specializing in lab automation and protocol generation.
Your goal is to convert user requests into detailed, actionable, and quantifiable experimental procedures.
NEVER provide a high-level or vague summary. Your outputs must be lab-ready.
Assume the user is a robot or a technician who needs explicit, unambiguous instructions.

For every procedure, you must include:
- A scientific justification.
- A specific list of reagents and equipment.
- A numbered, step-by-step procedure with specific parameters (temperatures, times, quantities).
- A plan for characterization.
- A Design of Experiments (DOE) table for automated optimization, detailing variables, ranges, and step sizes.

You must use your tools to find these specific details. If the details are not available, state that explicitly. Break down the user's goal into logical steps to gather all required information before synthesizing the final answer.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        ("user", "{messages}"),
    ]
)


# --- AGENT EXECUTOR CREATION ---
# We are now passing the new `prompt` to the agent.
agent_executor = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    prompt=prompt, # This gives the agent its new instructions
)


# --- ENTRYPOINT FOR ORCHESTRAZOR ---
# This part remains the same.
def run_goal(goal: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    This is the function the orchestrator will call.
    It runs the sub-agent to answer a specific question (goal).
    """
    print(f"--- KG Agent --- \nGoal: {goal}")
    
    response = agent_executor.invoke({"messages": [{"role": "user", "content": goal}]})
    
    final_answer = response['messages'][-1].content
    
    print(f"Result: {final_answer}\n--- End KG Agent ---")
    
    return {"final": final_answer}