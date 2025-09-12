from typing import Dict, Any

def cypher_from_task(task: Dict[str, Any]) -> str:
    q = (task.get("query") or "").lower()
    # Example: IAST CO2/N2 near ambient
    if "iast" in q and "co2" in q and "n2" in q:
        return """
        MATCH (m:Material {name:"Mg-MOF-74"})-[:HAS_MEASUREMENT]->(me:Measurement)-[:OF_PROPERTY]->(:Property {name:"IAST_selectivity_CO2_N2"})
        MATCH (me)-[:AT_CONDITION]->(c:Condition)
        WHERE c.temperature_K >= 273 AND c.temperature_K <= 323
          AND c.mixture = "CO2:0.15,N2:0.85"
        RETURN me.value AS selectivity, me.unit AS unit, c.temperature_K AS T, c.pressure_bar AS P,
               c.humidity_RH AS RH, me.method AS method, me.paper_id AS paper_id, me.line_no AS line_no
        ORDER BY T ASC LIMIT 50
        """
    # Fallback
    return "RETURN 0 AS noop"
