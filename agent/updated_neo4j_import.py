from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
import os

# ========== CONFIG ==========
load_dotenv()
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Hackathon"

# ========== FILE PATHS ==========
MATCHING_CSV = "hackathon_graph_data/final data/MOF_names_and_CSD_codes.csv"
SYNTHESIS_CSV = "hackathon_graph_data/final data/synthesis_extractions.csv"
APPLICATIONS_CSV = "hackathon_graph_data/final data/applications_filtered_v4.csv"
PROPERTIES_CSV = "hackathon_graph_data/final data/filtered_properties_v4.csv"

# ========== SETUP ==========
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def batch_run(session, query, rows, batch_size=1000):
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        session.run(query, {"rows": batch})

def create_indexes(session):
    queries = [
        "CREATE INDEX IF NOT EXISTS FOR (m:MOF) ON (m.refcode)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.name)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Property) ON (p.name, p.value)"
    ]
    for q in queries:
        session.run(q)

def import_matching(session):
    print("🔄 Importing MOF Names and DOIs...")
    df = pd.read_csv(MATCHING_CSV)
    rows = []
    for _, row in df.iterrows():
        names = str(row["MOF Name"]).split("<|>")
        refcode = row["CSD Ref Code"]
        doi = row["DOI"]
        if str(refcode).lower() in ["not provided", "not applicable"]:
            continue
        for name in names:
            if name.lower().strip() in ["not provided", "not applicable"]:
                continue
            rows.append({"refcode": refcode.strip(), "doi": doi, "name": name.strip()})
    query = '''
        UNWIND $rows AS row
        MERGE (m:MOF {refcode: row.refcode})
        MERGE (p:Paper {doi: row.doi})
        MERGE (m)-[:HAS_SOURCE]->(p)
    '''
    batch_run(session, query, rows)

def import_synthesis(session):
    print("🔄 Importing Synthesis Procedures...")
    df = pd.read_csv(SYNTHESIS_CSV)
    rows = df.to_dict("records")
    query = '''
        UNWIND $rows AS row
        MATCH (m:MOF {refcode: row.`CSD Ref Code`})
        MERGE (p:Paper {doi: row.Reference})
        CREATE (j:Text {
            type: "Synthesis Justification",
            justification: row.Justification
        })
        MERGE (j)-[:SYNTHESIZED_IN]->(m)
        MERGE (j)-[:HAS_SOURCE]->(p)
        FOREACH (ignore IN CASE WHEN row.Linker IS NOT NULL THEN [1] ELSE [] END |
            MERGE (l:Linker {name: row.Linker})
            MERGE (j)-[:USES]->(l)
        )
        FOREACH (ignore IN CASE WHEN row.`Metal Precursor` IS NOT NULL THEN [1] ELSE [] END |
            MERGE (mp:Precursor {name: row.`Metal Precursor`})
            MERGE (j)-[:USES]->(mp)
        )
        FOREACH (ignore IN CASE WHEN row.Solvent IS NOT NULL THEN [1] ELSE [] END |
            MERGE (sol:Solvent {name: row.Solvent})
            MERGE (j)-[:USES]->(sol)
        )
    '''
    batch_run(session, query, rows)

def import_applications(session):
    print("🔄 Importing Applications...")
    df = pd.read_csv(APPLICATIONS_CSV)
    df = df.rename(columns={"Source": "Reference"})
    rows = df.to_dict("records")
    query = '''
        UNWIND $rows AS row
        MATCH (m:MOF {refcode: row.`Ref Code`})
        MERGE (p:Paper {doi: row.Reference})
        CREATE (j:Text {
            type: "Application Justification",
            text: row.Justification
        })
        MERGE (a:Application {name: row.Application})
        MERGE (j)-[:MENTIONS]->(m)
        MERGE (j)-[:RECOMMENDS]->(a)
        MERGE (j)-[:HAS_SOURCE]->(p)
        MERGE (m)-[:IS_RECOMMENDED_FOR]->(a)
    '''
    batch_run(session, query, rows)

def import_properties(session):
    print("🔄 Importing Properties...")
    df = pd.read_csv(PROPERTIES_CSV)
    df = df.rename(columns={"Source": "Reference"})
    df = df[df["Value"].notna() & df["Property"].notna()]
    df["Units"] = df["Units"].fillna("")
    rows = df.to_dict("records")

    query = '''
        UNWIND $rows AS row
        MATCH (m:MOF {refcode: row.`Ref Code`})
        MERGE (p:Paper {doi: row.Reference})
        MERGE (prop:Property {
            name: row.Property,
            value: row.Value,
            units: row.Units
        })
        CREATE (j:Text {
            type: "Property Justification",
            text: row.Justification
        })
        MERGE (m)-[:HAS_PROPERTY]->(prop)
        MERGE (j)-[:SUPPORTS]->(prop)
        MERGE (j)-[:MENTIONS]->(m)
        MERGE (j)-[:HAS_SOURCE]->(p)
    '''
    batch_run(session, query, rows)

def run_all():
    with driver.session() as session:
        create_indexes(session)
        import_matching(session)
        import_synthesis(session)
        import_applications(session)
        import_properties(session)

if __name__ == "__main__":
    run_all()
    driver.close()