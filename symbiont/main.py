from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from truth_lens import TruthEngine

app = FastAPI(title="Symbiont: Truth Lens Node", version="1.0.0")
engine = TruthEngine()

class VerificationRequest(BaseModel):
    text: str
    source_url: str = None

@app.get("/")
def home():
    return {"status": "active", "node_type": "Truth Verifier", "version": "Alpha"}

@app.post("/verify")
def verify_claim(request: VerificationRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Run the Symbiont Analysis
    result = engine.analyze(request.text)
    
    return {
        "analysis": {
            "truth_score": result.truth_score,
            "trust_grade": get_grade(result.truth_score),
            "bias_level": result.bias_level,
            "network_consensus": f"{result.consensus_count} Nodes Verified",
        },
        "ecological_impact": {
            "rating": result.eco_cost,
        },
        "warnings": result.flags
    }

def get_grade(score):
    if score >= 90: return "A (Verified Fact)"
    if score >= 75: return "B (Likely True)"
    if score >= 50: return "C (Contested)"
    return "D (Unreliable/Opinion)"

# Instructions to run:
# uvicorn main:app --reload