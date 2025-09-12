import os, json, yaml
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from tools_tobias.cypher_tool import CypherTool
from tools_tobias.cypher_templates import cypher_from_task
from tools_tobias.vector_tool import VectorTool

CFG = yaml.safe_load(open("configs_tobias/settings.yaml"))
LLM = ChatOpenAI(model=CFG["models"]["chat"], temperature=0)

planner = (ChatPromptTemplate.from_messages([
    ("system", "You are a planning agent for Mg-MOF-74 CO2 selectivity in flue gas. Return ONLY JSON list of tasks."),
    ("user", "Goal: {goal}\nConstraints: {constraints}")
]) | LLM | StrOutputParser())

# instantiate tools
neo_pw = os.environ.get(CFG["neo4j"]["password_env"], "")
cy = CypherTool(CFG["neo4j"]["uri"], CFG["neo4j"]["user"], neo_pw, CFG["neo4j"].get("database"))
vec = VectorTool(CFG["vector"]["chroma_dir"])

def route(task: Dict[str, Any]) -> Dict[str, Any]:
    q = (task.get("query") or "").lower()
    use_graph = any(k in q for k in ["iast","isotherm","qst","selectivity","breakthrough","diffusion","pellet"])
    return {"task": task, "use_graph": use_graph, "use_vector": True}

router = RunnableLambda(route)
exec_graph = RunnableLambda(lambda r: cy.query(cypher_from_task(r["task"])) if r["use_graph"] else [])
exec_vec   = RunnableLambda(lambda r: vec.search(r["task"], k=5))

summ = (ChatPromptTemplate.from_messages([
    ("system", "Summarize for materials scientists. Keep numbers/units. Cite inline as [communityId] or [paper_id:line_no]."),
    ("user", "Task: {task}\nGraphRows: {graph}\nEvidenceLines: {vec}")
]) | LLM | StrOutputParser())

def run_goal(goal: str, constraints: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    constraints = constraints or {}
    tasks_json = planner.invoke({"goal": goal, "constraints": json.dumps(constraints)})
    try:
        tasks = json.loads(tasks_json)
        if isinstance(tasks, dict): tasks = [tasks]
    except Exception:
        tasks = [{"query": goal, "intent":"General", "conditions": constraints, "must_have":["line_citations"], "limit":20}]

    results = []
    for t in tasks:
        routed = router.invoke(t)
        g = exec_graph.invoke(routed)
        v = exec_vec.invoke(routed)
        s = summ.invoke({"task": json.dumps(t), "graph": json.dumps(g), "vec": json.dumps(v)})
        results.append({"task": t, "graph": g, "evidence": v, "summary": s})
    return results
