# tools/cypher_tool.py
from typing import Dict, Any, List, Tuple
from neo4j import GraphDatabase

class CypherTool:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database  # optional, if you use multi-db

    def close(self):
        self._driver.close()

    def query(self, cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with self._driver.session(database=self._database) as s:
            res = s.run(cypher, params or {})
            return [r.data() for r in res]

# —— trivial template mapping (replace with your schema)
def cypher_from_task(task: Dict[str, Any]) -> str:
    q = task["query"].lower()
    if "iast" in q and "co2" in q and "n2" in q:
        return """
        MATCH (m:Material {name:"Mg-MOF-74"})-[:HAS_MEASUREMENT]->(me:Measurement)-[:OF_PROPERTY]->(:Property {name:"IAST_selectivity_CO2_N2"})
        MATCH (me)-[:AT_CONDITION]->(c:Condition)
        WHERE c.temperature_K >= 273 AND c.temperature_K <= 323
          AND c.mixture = "CO2:0.15,N2:0.85"
        RETURN me.value AS selectivity, me.unit AS unit, c.temperature_K AS T, c.pressure_bar AS P,
               c.humidity_RH AS RH, me.method AS method, me.paper_id AS paper_id, me.line_no AS line_no
        ORDER BY T ASC LIMIT 50;
        """
    # Fallback: no-op (the vector tool will carry the task)
    return "RETURN 0 AS noop;"
