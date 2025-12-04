"""
Symbiont: Truth Lens Module
A decentralized-ready heuristic engine for Truth and Eco-Impact verification.
"""

import random
from textblob import TextBlob
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class VerificationResult:
    truth_score: float      # 0 to 100
    bias_level: float       # 0.0 (Objective) to 1.0 (Subjective)
    consensus_count: int    # How many nodes verified this
    eco_cost: str           # Low, Medium, High, Critical
    flags: List[str]        # Warnings (e.g., "High Subjectivity")

class EcoImpactCalculator:
    """
    Estimates ecological cost based on semantic analysis of resources/materials
    mentioned in the text.
    """
    def __init__(self):
        # Heuristic map: keyword -> impact score (1-10)
        self.impact_map = {
            "plastic": 8, "beef": 9, "oil": 10, "coal": 10, "shipping": 7,
            "flight": 9, "local": 2, "organic": 3, "solar": 2, "recycled": 2,
            "battery": 6, "lithium": 7, "cotton": 5, "fast fashion": 9
        }

    def calculate_impact(self, text: str) -> Dict:
        text_lower = text.lower()
        detected_items = []
        total_score = 0
        
        for keyword, score in self.impact_map.items():
            if keyword in text_lower:
                detected_items.append(keyword)
                total_score += score
        
        # Normalize score
        if not detected_items:
            return {"level": "Unknown", "score": 0, "items": []}
            
        avg_score = total_score / len(detected_items)
        
        if avg_score < 3: label = "Low (Eco-Friendly)"
        elif avg_score < 6: label = "Medium"
        elif avg_score < 8: label = "High"
        else: label = "Critical (High Carbon Debt)"
        
        return {"level": label, "score": round(avg_score, 1), "items": detected_items}

class FederatedVerifier:
    """
    Simulates querying a decentralized network of 'Symbiont' nodes 
    to verify facts against a distributed ledger.
    """
    def query_network(self, claim_hash: str) -> float:
        # In a real build, this would ping IPFS/Blockchain nodes.
        # Here we simulate network consensus with a stochastic model.
        # 1.0 = Full Consensus (True), 0.0 = Consensus (False)
        
        # Simulation: Random drift modified by "hash" stability
        simulated_consensus = random.uniform(0.6, 0.95)
        return simulated_consensus

class TruthEngine:
    def __init__(self):
        self.verifier = FederatedVerifier()
        self.eco_calc = EcoImpactCalculator()

    def analyze(self, text: str) -> VerificationResult:
        blob = TextBlob(text)
        
        # 1. Subjectivity Analysis (The Bias Filter)
        # 0.0 is very objective (fact), 1.0 is very subjective (opinion)
        subjectivity = blob.sentiment.subjectivity
        polarity = blob.sentiment.polarity
        
        # 2. Federated Verification (The Truth Check)
        # We simulate checking the claim against the network
        network_confidence = self.verifier.query_network(str(hash(text)))
        
        # 3. Calculate Truth Score
        # Formula: Start with Network Confidence, penalize for High Subjectivity
        # If text is highly subjective, it's likely an opinion, not a verified fact.
        penalty = 0
        if subjectivity > 0.5:
            penalty = (subjectivity - 0.5) * 40 # Max 20 point penalty
            
        truth_score = (network_confidence * 100) - penalty
        truth_score = max(0, min(100, truth_score)) # Clamp between 0-100
        
        # 4. Eco Scan
        eco_data = self.eco_calc.calculate_impact(text)
        
        # 5. Generate Flags
        flags = []
        if subjectivity > 0.6:
            flags.append("High Subjectivity Detected (Opinion/Emotion)")
        if eco_data['score'] > 7:
            flags.append(f"High Environmental Impact: {', '.join(eco_data['items'])}")
        if truth_score < 50:
            flags.append("Low Network Consensus (Unverified)")

        return VerificationResult(
            truth_score=round(truth_score, 1),
            bias_level=round(subjectivity, 2),
            consensus_count=random.randint(120, 5000), # Simulating active nodes
            eco_cost=eco_data['level'],
            flags=flags
        )