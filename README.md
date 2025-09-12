# MOF Research Agent

This project is used for the LLM Hackathon for Applications in Materials and Chemistry. It's a LangGraph-powered autonomous agent designed to assist researchers by planning and executing complex queries against a knowledge base of scientific summaries on Metal-Organic Frameworks (MOFs).

## ✨ Features

* **Planner-Executor Architecture:** The agent first creates a multi-step plan to solve a user's high-level goal and then executes each step in a loop.
* **Retrieval-Augmented Generation (RAG):** The agent's knowledge comes from a custom vector database of MOF community summaries, ensuring answers are grounded in specific data.
* **Source Citation:** The agent is designed to cite its sources, referencing the specific `communityId` from the knowledge base that it used to formulate its answer.
* **Observability:** Integrated with LangSmith for detailed tracing and debugging of the agent's reasoning process.

---

## 🏛️ System Architecture

The agent operates on a multi-level graph defined in LangGraph. The high-level execution flow is as follows:

**Execution Flow:**
`demo.py` -> `Orchestrator Graph` -> `Planner Node` -> `Executor Node (Loop)` -> `Synthesizer Node` -> Final Answer

### Key Files and Their Roles

This diagram shows the main project structure. Below is an explanation of each key component.