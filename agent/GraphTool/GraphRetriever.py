import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()


class MOFRagSystem:
    """MOF Research Assistant using RAG over community summaries."""
    
    def __init__(self, summaries_file: str = "community_summaries_cleaned.jsonl", 
                 persist_dir: str = "agent/GraphTool/chroma_db_mof_summaries"):
        self.summaries_file = summaries_file
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_system()
    
    def _load_summaries(self) -> List[Dict[str, Any]]:
        """Load community summaries from JSONL file."""
        summaries = []
        if not os.path.exists(self.summaries_file):
            raise FileNotFoundError(f"Summaries file not found: {self.summaries_file}")
            
        with open(self.summaries_file, "r") as f:
            for line in f:
                summaries.append(json.loads(line))
        
        print(f"Loaded {len(summaries)} summaries")
        return summaries
    
    def _create_documents(self, summaries: List[Dict[str, Any]]) -> List[Document]:
        """Convert summaries to LangChain documents with MOF name cutoff."""
        docs = []
        for s in summaries:
            all_mofs = s.get("mofs", [])
            n_mofs = len(all_mofs)

            # Include MOF names only for communities with <= 10 MOFs
            if n_mofs <= 10:
                mof_text = f"MOFs: {', '.join(all_mofs)}"
            else:
                mof_text = "MOFs: [list omitted due to large community size]"

            text = f"""Community ID: {s['communityId']}
{mof_text}
Summary: {s['summary']}""".strip()

            docs.append(Document(
                page_content=text,
                metadata={
                    "communityId": s["communityId"],
                }
            ))
        
        return docs
    
    def _setup_vectorstore(self, docs: List[Document]) -> None:
        """Create or load the vector store."""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Check if persistent directory exists and has data
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=self.persist_dir
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=self.persist_dir
            )
            self.vectorstore.persist()
    
    def _setup_qa_chain(self) -> None:
        """Initialize the RAG QA chain."""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert chemist specializing in Metal-Organic Frameworks (MOFs). "
             "You have access to curated community summaries that group MOFs by shared properties, "
             "applications, and structural features. "
             "Your task is to use these summaries as context to infer scientific insights and answer the user's question.\n\n"
             "Guidelines:\n"
             "- Identify and explain patterns (e.g., why certain MOFs are clustered together).\n"
             "- Make inferences about application-property-structure relationships.\n"
             "- Be concise, factual, and use technical terminology appropriate for a materials scientist.\n"
             "- If the context does not fully answer the question, say so explicitly and suggest what is known.\n"
             "- Do not mention community summaries or data sources in your response.\n"
            ),
            ("user", "Question: {question}\n\nContext:\n{context}")
        ])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o", temperature=0.2),  # Changed from gpt-5 to gpt-4o
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": rag_prompt,
                "document_variable_name": "context"
            }
        )
    
    def _initialize_system(self) -> None:
        """Initialize the complete RAG system."""
        try:
            summaries = self._load_summaries()
            docs = self._create_documents(summaries)
            self._setup_vectorstore(docs)
            self._setup_qa_chain()
            print("MOF RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing MOF RAG system: {e}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the MOF RAG system."""
        if not self.qa_chain:
            raise RuntimeError("RAG system not properly initialized")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "community_id": doc.metadata.get("communityId"),
                        "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {
                "answer": f"Error querying MOF database: {str(e)}",
                "sources": []
            }


# Global instance for tool usage
_mof_rag_system = None

def get_mof_rag_system() -> MOFRagSystem:
    """Get or create the global MOF RAG system instance."""
    global _mof_rag_system
    if _mof_rag_system is None:
        _mof_rag_system = MOFRagSystem()
    return _mof_rag_system


def query_mof_database(question: str) -> str:
    """
    Query the MOF research database for information about Metal-Organic Frameworks.
    
    This tool provides expert-level insights about MOF synthesis, properties, applications,
    and structure-property relationships based on curated community summaries.
    
    Args:
        question: A question about MOFs, their properties, synthesis, or applications
        
    Returns:
        Expert analysis and recommendations based on MOF research data
    """
    try:
        rag_system = get_mof_rag_system()
        result = rag_system.query(question)

        response = result["answer"]
        if result["sources"]:
            response += "\n\n[Based on analysis of relevant MOF research communities]"
        return response

    except Exception as e:
        return f"Unable to query MOF database: {str(e)}"



# Example usage and testing
if __name__ == "__main__":
    # Test the tool
    test_query = "How can I synthesize MOF mg-mof-74 to be more selective to CO2 in flue gas?"
    result = query_mof_database(test_query)
    print("Query:", test_query)
    print("Response:", result)