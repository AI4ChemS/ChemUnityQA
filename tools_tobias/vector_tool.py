from typing import Dict, Any, List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorTool:
    def __init__(self, persist_dir: str):
        self.emb = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store = Chroma(embedding_function=self.emb, persist_directory=persist_dir)
        self.retriever = self.store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def search(self, task: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        q = task.get("query", "")
        docs = self.retriever.get_relevant_documents(q)
        out = []
        for d in docs[:k]:
            out.append({
                "text": d.page_content,
                "communityId": d.metadata.get("communityId"),
                # keep placeholders for future line-level cites:
                "paper_id": d.metadata.get("paper_id"),
                "line_no": d.metadata.get("line_no"),
            })
        return out
