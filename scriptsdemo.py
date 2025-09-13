# demo.py

import langchain
langchain.debug = True # This will show you the agent's thought process

from agent.orchestrator_graph import run_orchestrator

if __name__ == "__main__":
    # LangGraph will generate subqueries for this basic prompt.
    question = """
Generate three distinct, lab-ready experimental protocols to increase the CO2 selectivity of Mg-MOF-74 in flue gas. Cite the specific community IDs from the knowledge graph that you used to find the answer.
"""
    
    answer = run_orchestrator(question)
    
    print("\n\n" + "="*20)
    print("   FINAL SYNTHESIS")
    print("="*20)
    print(answer)