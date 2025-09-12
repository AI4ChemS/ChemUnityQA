from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
from openai import OpenAI
from tqdm import tqdm
import json
import time

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

client = OpenAI(api_key=OPENAI_API_KEY)

class MOFGraphRAG:
    def __init__(self, graph, llm_client):
        self.graph = graph
        self.client = llm_client
    
    def create_graph_projection(self):
        """Create graph projection for community detection"""
        print("Creating graph projection...")
        
        # Drop existing projection if it exists
        try:
            self.graph.query("CALL gds.graph.drop('mof_shared_features')")
        except:
            pass
        
        # Create new projection
        result = self.graph.query("""
        CALL gds.graph.project(
            'mof_shared_features',
            'MOF',
            {
                SIMILAR_TO: {orientation: 'UNDIRECTED'},
                SHARES_PROPERTY: {orientation: 'UNDIRECTED'},
                SAME_APPLICATION: {orientation: 'UNDIRECTED'}
            }
        )
        """)
        print(f"Graph projection created: {result}")
        return result
    
    def run_leiden_clustering(self):
        """Run Leiden clustering algorithm"""
        print("Running Leiden clustering...")
        
        result = self.graph.query("""
        CALL gds.leiden.write('mof_shared_features', {
            writeProperty: 'community',
            includeIntermediateCommunities: true,
            maxLevels: 4
        })
        YIELD communityCount, modularity
        """)
        print(f"Leiden clustering completed: {result}")
        return result
    
    def create_community_nodes(self, max_level=3):
        """Create community nodes for each hierarchy level"""
        print("Creating community hierarchy...")
        
        # Create communities for each level
        for level in range(max_level + 1):
            self.graph.query(f"""
            MATCH (m:MOF)
            WITH DISTINCT m.communityLevel{level} AS cid
            WHERE cid IS NOT NULL
            MERGE (c:Community {{id: cid, level: {level}}})
            """)
        
        # Connect MOFs to level 0 communities
        self.graph.query("""
        MATCH (m:MOF)
        WHERE m.communityLevel0 IS NOT NULL
        WITH m, m.communityLevel0 AS cid
        MATCH (c:Community {id: cid, level: 0})
        MERGE (m)-[:IN_COMMUNITY]->(c)
        """)
        
        # Create hierarchical relationships
        for level in range(1, max_level + 1):
            self.graph.query(f"""
            MATCH (m:MOF)
            WHERE m.communityLevel{level-1} IS NOT NULL AND m.communityLevel{level} IS NOT NULL
            WITH DISTINCT m.communityLevel{level-1} AS child_id, m.communityLevel{level} AS parent_id
            MATCH (child:Community {{id: child_id, level: {level-1}}})
            MATCH (parent:Community {{id: parent_id, level: {level}}})
            MERGE (child)-[:SUBCOMMUNITY_OF]->(parent)
            """)
        
        print("Community hierarchy created")
    
    def get_community_info(self, level=0):
        """Extract community information including MOFs, applications, properties, and evidence"""
        print(f"Extracting community information for level {level}...")
        
        community_info = self.graph.query(f"""
        MATCH (c:Community {{level: {level}}})<-[:IN_COMMUNITY]-(m:MOF)
        OPTIONAL MATCH (t:Text)-[:MENTIONS|SUPPORTS|SYNTHESIZED_IN]->(m)
        OPTIONAL MATCH (m)-[:HAS_PROPERTY]->(prop:Property)
        OPTIONAL MATCH (m)-[:IS_RECOMMENDED_FOR]->(a:Application)
        OPTIONAL MATCH (m)-[:HAS_LINKER]->(linker:Linker)
        OPTIONAL MATCH (m)-[:HAS_NODE]->(node:Node)
        
        WITH c, 
             collect(DISTINCT m.refcode) AS mofs,
             collect(DISTINCT a.name) AS applications,
             collect(DISTINCT prop.name + ': ' + toString(prop.value) + ' ' + COALESCE(prop.units, '')) AS properties,
             collect(DISTINCT linker.name) AS linkers,
             collect(DISTINCT node.name) AS nodes,
             [txt IN collect(DISTINCT t.text) WHERE txt IS NOT NULL AND txt <> '' AND txt <> '<no justification>'] AS justifications
        
        RETURN c.id AS communityId,
               c.level AS level,
               size(mofs) AS mof_count,
               mofs[..10] AS sample_mofs,  // Limit for prompt size
               applications[..10] AS applications,
               properties[..15] AS properties,
               linkers[..10] AS linkers,
               nodes[..10] AS nodes,
               justifications[..5] AS evidence_snippets
        ORDER BY mof_count DESC
        """)
        
        print(f"Found {len(community_info)} communities at level {level}")
        return community_info
    
    def create_community_summary_prompt(self, comm):
        """Create a detailed prompt for community summarization"""
        mofs = ", ".join([str(m) for m in comm["sample_mofs"] if m]) or "None"
        apps = ", ".join([str(a) for a in comm["applications"] if a]) or "None"
        props = "; ".join([str(p) for p in comm["properties"] if p]) or "None"
        linkers = ", ".join([str(l) for l in comm["linkers"] if l]) or "None"
        nodes = ", ".join([str(n) for n in comm["nodes"] if n]) or "None"
        evidence = " | ".join([str(e) for e in comm["evidence_snippets"] if e]) or "None"
        
        return f"""You are a scientific expert in Metal-Organic Frameworks (MOFs). Analyze this MOF community and provide a comprehensive summary.

Community Details:
- Community ID: {comm['communityId']}
- Level: {comm['level']}
- Number of MOFs: {comm['mof_count']}
- Sample MOFs: {mofs}
- Applications: {apps}
- Key Properties: {props}
- Common Linkers: {linkers}
- Metal Nodes: {nodes}
- Research Evidence: {evidence}

Please provide a structured summary that includes:

1. **Community Theme**: What unifies these MOFs? (chemical composition, structure type, applications)
2. **Key Characteristics**: Distinctive structural or chemical features
3. **Primary Applications**: Main use cases and performance areas
4. **Notable Properties**: Important physical/chemical properties
5. **Research Insights**: Key findings from the evidence snippets

Requirements:
- Be scientifically accurate and use appropriate MOF terminology
- Keep the summary concise but informative (4-8 sentences)
- Highlight what makes this community unique
- If DOIs or specific studies are mentioned in evidence, reference them
- Focus on the most significant patterns and applications

Format as a cohesive paragraph, not bullet points."""

    def generate_community_summaries(self, community_info, batch_size=5):
        """Generate LLM-powered summaries for communities"""
        print(f"Generating summaries for {len(community_info)} communities...")
        
        summaries = []
        
        for i in tqdm(range(0, len(community_info), batch_size), desc="Processing community batches"):
            batch = community_info[i:i+batch_size]
            
            for comm in batch:
                try:
                    prompt = self.create_community_summary_prompt(comm)
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4",  # Use gpt-4 or gpt-3.5-turbo
                        messages=[
                            {"role": "system", "content": "You are a materials science expert specializing in Metal-Organic Frameworks (MOFs). Provide clear, scientifically accurate summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    summaries.append({
                        "communityId": comm["communityId"],
                        "level": comm["level"],
                        "mof_count": comm["mof_count"],
                        "summary": summary
                    })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error generating summary for community {comm['communityId']}: {e}")
                    summaries.append({
                        "communityId": comm["communityId"],
                        "level": comm["level"],
                        "mof_count": comm["mof_count"],
                        "summary": f"Error generating summary: {str(e)}"
                    })
        
        print(f"Generated {len(summaries)} summaries")
        return summaries
    
    def store_summaries_in_graph(self, summaries):
        """Store generated summaries back in Neo4j"""
        print("Storing summaries in graph...")
        
        for summary in tqdm(summaries, desc="Storing summaries"):
            self.graph.query("""
            MATCH (c:Community {id: $community_id, level: $level})
            SET c.summary = $summary,
                c.mof_count = $mof_count,
                c.summary_generated = datetime()
            """, {
                "community_id": summary["communityId"],
                "level": summary["level"],
                "summary": summary["summary"],
                "mof_count": summary["mof_count"]
            })
        
        print("Summaries stored successfully")
    
    def create_hierarchical_summaries(self, max_level=3):
        """Create summaries for higher-level communities based on their subcommunities"""
        print("Creating hierarchical summaries...")
        
        for level in range(1, max_level + 1):
            print(f"Processing level {level} communities...")
            
            higher_communities = self.graph.query(f"""
            MATCH (parent:Community {{level: {level}}})<-[:SUBCOMMUNITY_OF]-(child:Community)
            WHERE child.summary IS NOT NULL
            WITH parent,
                 collect(child.summary) AS child_summaries,
                 count(child) AS child_count
            RETURN parent.id AS communityId,
                   parent.level AS level,
                   child_summaries,
                   child_count
            """)
            
            for comm in higher_communities:
                try:
                    # Create prompt for hierarchical summary
                    child_summaries = "\n\n".join([f"- {summary}" for summary in comm["child_summaries"]])
                    
                    prompt = f"""You are analyzing a higher-level MOF research community composed of {comm['child_count']} subcommunities.

Subcommunity Summaries:
{child_summaries}

Please create a unified summary for this higher-level community (Level {level}) that:
1. Identifies the overarching themes connecting all subcommunities
2. Highlights the diversity within this broader community
3. Synthesizes the main applications and properties across subcommunities
4. Explains what makes this a coherent research area

Keep it concise (3-5 sentences) and focus on the big picture themes that unite these subcommunities."""

                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a materials science expert creating hierarchical summaries of MOF research communities."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    
                    # Store hierarchical summary
                    self.graph.query("""
                    MATCH (c:Community {id: $community_id, level: $level})
                    SET c.summary = $summary,
                        c.child_count = $child_count,
                        c.hierarchical_summary_generated = datetime()
                    """, {
                        "community_id": comm["communityId"],
                        "level": comm["level"],
                        "summary": summary,
                        "child_count": comm["child_count"]
                    })
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error generating hierarchical summary for community {comm['communityId']}: {e}")
        
        print("Hierarchical summaries completed")
    
    def run_full_pipeline(self):
        """Run the complete GraphRAG pipeline"""
        print("Starting MOF GraphRAG pipeline...")
        
        # Step 1: Create graph projection
        self.create_graph_projection()
        
        # Step 2: Run community detection
        self.run_leiden_clustering()
        
        # Step 3: Create community hierarchy
        self.create_community_nodes()
        
        # Step 4: Generate summaries for level 0 communities
        community_info = self.get_community_info(level=0)
        summaries = self.generate_community_summaries(community_info)
        self.store_summaries_in_graph(summaries)
        
        # Step 5: Create hierarchical summaries
        self.create_hierarchical_summaries()
        
        print("MOF GraphRAG pipeline completed successfully!")
        
        return summaries

# Usage example
if __name__ == "__main__":
    # Initialize the GraphRAG system
    mof_graphrag = MOFGraphRAG(graph, client)
    
    # Run the complete pipeline
    summaries = mof_graphrag.run_full_pipeline()
    
    # Display some example summaries
    print("\n=== Sample Community Summaries ===")
    for summary in summaries[:3]:
        print(f"\nCommunity {summary['communityId']} ({summary['mof_count']} MOFs):")
        print(summary['summary'])
        print("-" * 80)