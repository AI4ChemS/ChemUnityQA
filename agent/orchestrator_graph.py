# agent/orchestrator_graph.py
# --- FINAL VERSION ---

import os, json, yaml, re
from typing import TypedDict, Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- Shared State ---
class OrchestratorState(TypedDict):
    user_input: str
    goal: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    final: str

# --- Node-specific imports ---
from agent.GraphTool.kg_agent import run_goal as run_kg_graph

# --- Nodes ---
def parse_node(state: OrchestratorState):
    """Takes the user input and sets it as the initial goal."""
    print("--- PARSE ---")
    return {"goal": state["user_input"]}

def planner_node(state: OrchestratorState):
    """Creates a plan to solve the user's goal."""
    print("--- PLANNER ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert materials science researcher. Your job is to create a concise, high-level, step-by-step plan to answer the user's goal. "
         "Create a plan with **no more than 5-7 steps**. "
         "Each step should be a complete research task that can be delegated to an assistant. For example, a single step should be 'Find a complete, lab-ready protocol for amine functionalization of Mg-MOF-74', not broken down into smaller pieces. "
         "Respond with a numbered list of steps and nothing else."),
        ("user", "User's Goal: {goal}")
    ])
    planner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | planner_llm
    result = chain.invoke({"goal": state["goal"]})
    
    # UPDATED: More robust parsing to handle complex list formats
    plan_steps = re.findall(r"^\d+\.\s.*", result.content, re.MULTILINE)
    print(f"Plan: {plan_steps}")
    return {"plan": plan_steps}

def executor_node(state: OrchestratorState):
    """Executes a single step of the plan."""
    print("--- EXECUTOR ---")
    if not state["plan"]:
        raise ValueError("No plan to execute.")
    step = state["plan"][0]
    remaining_plan = state["plan"][1:]
    print(f"Executing step: {step}")
    result_dict = run_kg_graph(goal=step, constraints={})
    result = result_dict.get("final", '{"answer": "No result found.", "sources": []}')
    new_past_steps = state.get("past_steps", []) + [(step, result)]
    return {"plan": remaining_plan, "past_steps": new_past_steps}

def synthesizer_node(state: OrchestratorState):
    """Synthesizes the results of all steps into a final answer, citing sources."""
    print("--- SYNTHESIZER ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert materials science writer. Your job is to synthesize the results of a research plan into a final, comprehensive report. "
         "The user's original goal and a list of completed steps are provided below. Each step's result is a JSON object containing an 'answer' and a list of 'sources' (community IDs from the knowledge graph). "
         "You MUST use the information from the 'answer' fields to construct your report. "
         "At the end of your report, you MUST include the references to the MOF knowledge graph by their original refcode"),
        ("user", "Original Goal: {goal}\n\nCompleted Steps and Results:\n{past_steps}")
    ])
    synthesizer_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    chain = prompt | synthesizer_llm
    past_steps_str = "\n\n".join([f"Step: {step}\nResult: {result}" for step, result in state["past_steps"]])
    result = chain.invoke({"goal": state["goal"], "past_steps": past_steps_str})
    return {"final": result.content}

def should_continue(state: OrchestratorState) -> str:
    """Determines whether to continue the loop or end."""
    if state["plan"]:
        return "continue"
    else:
        return "end"

# --- Build Graph ---
g = StateGraph(OrchestratorState)

g.add_node("parse", parse_node)
g.add_node("planner", planner_node)
g.add_node("executor", executor_node)
g.add_node("synthesizer", synthesizer_node)

g.set_entry_point("parse")
g.add_edge("parse", "planner")
g.add_edge("planner", "executor")
g.add_conditional_edges(
    "executor",
    should_continue,
    {"continue": "executor", "end": "synthesizer"}
)
g.add_edge("synthesizer", END)

app = g.compile()

def run_orchestrator(user_input: str):
    initial_state = {
        "user_input": user_input, "goal": "", "plan": [], "past_steps": [], "final": ""
    }
    config = {"recursion_limit": 100}
    out = app.invoke(initial_state, config=config)
    return out["final"]