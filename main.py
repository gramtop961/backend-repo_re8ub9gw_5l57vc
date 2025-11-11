import os
from typing import List, Dict, Set, Tuple, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Knowledge Base (sample)
# ---------------------------
class Rule(BaseModel):
    antecedents: List[str]
    consequent: str
    description: Optional[str] = None

# Different rule sets for forward vs backward reasoning
# Forward rules: broader, more permissive for exploratory hypothesis generation
FORWARD_RULES: List[Rule] = [
    Rule(antecedents=["battery_low"], consequent="power_unstable", description="Low battery can cause unstable power"),
    Rule(antecedents=["power_unstable"], consequent="system_restarts", description="Unstable power can trigger restarts"),
    Rule(antecedents=["no_wifi", "router_off"], consequent="network_down", description="No WiFi and router off implies network down"),
    Rule(antecedents=["network_down"], consequent="cannot_sync", description="If the network is down, syncing fails"),
    # Fault hypotheses (prefix fault_)
    Rule(antecedents=["power_unstable"], consequent="fault_power_supply", description="Unstable power suggests power supply fault"),
    Rule(antecedents=["battery_low", "charging_not_working"], consequent="fault_battery", description="Low battery + charging not working suggests battery fault"),
    Rule(antecedents=["network_down"], consequent="fault_network", description="Network down suggests network fault"),
]

# Backward rules: stricter, require stronger evidence for a proof
BACKWARD_RULES: List[Rule] = [
    # Derivations for intermediate states
    Rule(antecedents=["battery_low"], consequent="power_unstable", description="Low battery can cause unstable power"),
    Rule(antecedents=["mains_fluctuation"], consequent="power_unstable", description="Mains fluctuation can cause unstable power"),
    Rule(antecedents=["power_unstable"], consequent="system_restarts", description="Unstable power can trigger restarts"),
    Rule(antecedents=["interference", "weak_signal"], consequent="no_wifi", description="Interference and weak signal cause Wi‑Fi loss"),
    Rule(antecedents=["no_wifi", "router_off"], consequent="network_down", description="Router off with no Wi‑Fi implies network is down"),
    Rule(antecedents=["network_down"], consequent="cannot_sync", description="No network means syncing fails"),
    # Fault hypotheses (require stronger antecedents)
    Rule(antecedents=["power_unstable", "system_restarts"], consequent="fault_power_supply", description="Unstable power AND restarts indicate power supply fault"),
    Rule(antecedents=["battery_low", "charging_not_working", "old_battery"], consequent="fault_battery", description="Low, not charging, and aged battery indicates battery fault"),
    Rule(antecedents=["no_wifi", "router_off", "cannot_sync"], consequent="fault_network", description="No Wi‑Fi, router off, and cannot sync indicates network fault"),
]

FAULT_PREFIX = "fault_"

# ---------------------------
# Inference Engines
# ---------------------------

def forward_chain(facts: Set[str], rules: List[Rule]) -> Tuple[Set[str], List[Dict]]:
    """Simple forward chaining for propositional Horn rules.
    Returns (all_derived_facts, trace)
    trace: list of {rule, new_fact} applied in order
    """
    known = set(facts)
    trace = []
    applied = True
    while applied:
        applied = False
        for r in rules:
            # If all antecedents are known and consequent not yet known, derive it
            if all(a in known for a in r.antecedents) and r.consequent not in known:
                known.add(r.consequent)
                trace.append({
                    "antecedents": r.antecedents,
                    "consequent": r.consequent,
                    "description": r.description,
                })
                applied = True
    return known, trace


def backward_chain(goal: str, facts: Set[str], rules: List[Rule],
                   visited: Optional[Set[str]] = None) -> Tuple[bool, List[Dict]]:
    """Depth-first backward chaining to prove goal from facts using rules.
    Returns (provable, proof_steps)
    proof_steps is a list with entries describing how goals were satisfied.
    """
    if visited is None:
        visited = set()

    # If goal is already a known fact
    if goal in facts:
        return True, [{"goal": goal, "type": "given"}]

    if goal in visited:
        return False, [{"goal": goal, "type": "cycle"}]

    visited.add(goal)

    # Consider rules that conclude the goal
    candidate_rules = [r for r in rules if r.consequent == goal]
    for r in candidate_rules:
        subproof = []
        all_ok = True
        for subgoal in r.antecedents:
            ok, proof = backward_chain(subgoal, facts, rules, visited)
            subproof.extend(proof)
            if not ok:
                all_ok = False
                break
        if all_ok:
            # Entire rule satisfied
            step = {
                "goal": goal,
                "type": "inferred",
                "using": {
                    "antecedents": r.antecedents,
                    "consequent": r.consequent,
                    "description": r.description,
                },
                "subproof": subproof,
            }
            return True, [step]

    # If no rule can prove it and it's not a given fact
    return False, [{"goal": goal, "type": "not-provable"}]


# ---------------------------
# API Models
# ---------------------------
class FactsRequest(BaseModel):
    facts: List[str]

class BackwardRequest(BaseModel):
    facts: List[str]
    goal: str

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Symbolic Fault Diagnosis API"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Simple health endpoint for this activity (database optional)."""
    return {
        "backend": "✅ Running",
        "database": "ℹ️ Not used for this activity",
    }

@app.get("/rules")
def get_rules():
    return {
        "forward_rules": [r.model_dump() for r in FORWARD_RULES],
        "backward_rules": [r.model_dump() for r in BACKWARD_RULES],
        "fault_prefix": FAULT_PREFIX,
    }

@app.post("/diagnose/forward")
def diagnose_forward(req: FactsRequest):
    input_facts = set(a.strip() for a in req.facts if a and a.strip())
    all_facts, trace = forward_chain(input_facts, FORWARD_RULES)
    faults = sorted([f for f in all_facts if f.startswith(FAULT_PREFIX)])
    return {
        "input_facts": sorted(list(input_facts)),
        "derived_facts": sorted(list(all_facts - input_facts)),
        "trace": trace,
        "faults": faults,
    }

@app.post("/diagnose/backward")
def diagnose_backward(req: BackwardRequest):
    input_facts = set(a.strip() for a in req.facts if a and a.strip())
    goal = req.goal.strip()
    provable, proof = backward_chain(goal, input_facts, BACKWARD_RULES)
    return {
        "goal": goal,
        "facts": sorted(list(input_facts)),
        "provable": provable,
        "proof": proof,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
