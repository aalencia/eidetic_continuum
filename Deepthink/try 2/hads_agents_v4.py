# eidetic_continuum/hads_agents_v4.py
"""
Enhanced HADS Agents v4
Modular components (LLM, Embedder, VectorStore) updated for:
1. Compatibility with ECE v2's Critic/Refine agents.
2. Full integration with HADS v4's Event Bus for auditability.
"""

import json
import hashlib
from typing import List, Dict, Any
import time
import uuid

# Import models from the engine for type hinting/mocking
try:
    from hads_engine_v4 import HILResponse, HILAction, DistortionScore, HADSExecutionResult, ECEConfig, StructuredMemory, BeliefStability, BeliefNode
except ImportError:
    # Minimal fallback structure for standalone testing
    class HILResponse: pass
    class HILAction: ACCEPT, REJECT, MODIFY, DEFER = 1, 2, 3, 4
    class DistortionScore: pass
    class HADSExecutionResult: pass
    class ECEConfig: pass
    class StructuredMemory: pass
    class BeliefStability: CORE, WORKING, EPHEMERAL = 'core', 'working', 'ephemeral'
    class BeliefNode: pass


class HADSEnhancedMockEmbedder:
    """Deterministic embedder with semantic simulation for HADS"""
    
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.semantic_boosters = {
            "core": ["fundamental", "essential", "critical", "non-negotiable"],
            "compliance": ["compliance", "regulation", "policy", "requirement"],
            "risk": ["risk", "danger", "violation", "prohibited"]
        }
    
    def embed(self, text: str) -> List[float]:
        # Use hash for determinism but create varied embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Create vector from hash
        vector = []
        for i in range(self.dimensions):
            chunk = text_hash[i * 2: (i + 1) * 2]
            val = int(chunk, 16) / 255.0  # Normalize to 0-1
            vector.append(val)
        
        # Apply semantic boosting based on content
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.semantic_boosters["core"]):
            vector[0] *= 1.5
        if any(word in text_lower for word in self.semantic_boosters["compliance"]):
            vector[1] *= 1.3
        if any(word in text_lower for word in self.semantic_boosters["risk"]):
            vector[2] *= 1.4
        
        # Normalize
        norm = sum(x * x for x in vector) ** 0.5
        return [x / (norm + 1e-12) for x in vector]


class HADSInMemoryVectorStore:
    """Enhanced with HADS event publishing"""
    
    def __init__(self, event_bus=None):
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.event_bus = event_bus
    
    def add_document(self, id: str, text: str, embedding: List[float], 
                    metadata: Dict[str, Any]):
        self.docs[id] = {
            "id": id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }
    
    def query_by_embedding(self, embedding: List[float], k: int = 5,
                          trace_id: str = "") -> List[Dict[str, Any]]:
        # Simplified query logic
        def sim(doc_emb):
            norm_a = sum(a * a for a in embedding) ** 0.5
            norm_b = sum(b * b for b in doc_emb) ** 0.5
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 0.0
            return sum(a * b for a, b in zip(embedding, doc_emb)) / (norm_a * norm_b)
        
        scored = [(sim(d["embedding"]), d) for d in self.docs.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in scored[:k]]


class HADSEnhancedMockLLM:
    """LLM with HADS-specific reasoning and ECE refinement methods"""
    
    def __init__(self):
        self.hil_accept = False
        self.hil_action = HILAction.ACCEPT
        self.hil_modified_text = None
        self.hil_confidence = 1.0
        self.reasoning_log: List[Dict] = []
    
    def generate(self, prompt: str) -> str:
        """Entry point for LLM generation/agent calls"""
        try:
            p = json.loads(prompt)
        except Exception:
            return self._default_response(prompt)
        
        role = p.get("role")
        
        if role == "hads_agent":
            return self._hads_agent_response(p)
        
        return self._default_response(prompt)

    # --- ECE Refinement Agent Methods (Called by Engine during the loop) ---
    
    def critic_response(self, interpretation: str, context_docs: List[Dict]) -> str:
        """Simulates the Critic Agent (ECE v2)"""
        critique = "Analysis: "
        if any("privacy" in doc.get("text", "").lower() for doc in context_docs):
            critique += "Detected sensitive context. Interpretation needs careful alignment with 'privacy' principles."
        elif "REFINED" in interpretation:
            critique += "Minor semantic drift remaining. Suggest using more technical language."
        else:
            critique += "Interpretation seems okay but needs to be less ambiguous."
        return critique
    
    def refine_response(self, interpretation: str, critique: str, context_docs: List[Dict]) -> str:
        """Simulates the Refine Agent (ECE v2)"""
        refined = interpretation
        
        if "privacy" in critique.lower():
            refined = refined.replace("user data", "anonymized user data")
        
        if "[REFINED]" not in refined:
            refined += " [REFINED]"
        else:
            refined = refined.replace("[REFINED]", "[REFINED-RECURSE]")
            
        return refined
    
    # --- HADS Agent Methods ---

    def _hads_agent_response(self, prompt: Dict) -> str:
        """HADS agent response with constraint awareness (Clear the Haze)"""
        input_text = prompt.get("input", "")
        constraints = prompt.get("constraints", {})
        
        reasoning_steps = []
        constraints_applied = []
        
        # Check must_avoid constraints
        if "violate_compliance_policies" in constraints.get("must_avoid", []):
            reasoning_steps.append("Acknowledging compliance guardrail.")
            constraints_applied.append("compliance_guardrail")
        
        # Check validation requirements
        if "human_approval_required" in constraints.get("validation_requirements", []):
            reasoning_steps.append("Flagging per high-value review policy.")
            constraints_applied.append("human_escalation_required")
        
        interpretation = f"HADS Interpretation: {input_text[:80]}"
        if constraints_applied:
            interpretation += f" [CONSTRAINTS: {', '.join(constraints_applied)}]"
        
        confidence = 0.7
        if constraints_applied:
            confidence = 0.5
        
        return json.dumps({
            "interpretation": interpretation,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "constraints_applied": constraints_applied,
            "trace_id": prompt.get("trace_id", str(uuid.uuid4()))
        })
    
    def _default_response(self, prompt: str) -> str:
        return json.dumps({
            "interpretation": str(prompt),
            "delta_summary": str(prompt)[:200],
            "confidence": 0.5,
        })
    
    def set_hil_behavior(
        self,
        accept: bool = False,
        action: HILAction = HILAction.ACCEPT,
        modified_text: str = None,
        confidence: float = 1.0,
        notes: str = "",
    ):
        """Configure HIL behavior for testing"""
        self.hil_accept = accept
        self.hil_action = action
        self.hil_modified_text = modified_text
        self.hil_confidence = confidence
        self.hil_notes = notes


class HADSTestHarness:
    """Enhanced test harness for HADS validation"""
    
    @staticmethod
    def create_hads_test_system(hil_accept: bool = False):
        """Create complete HADS test system"""
        from hads_engine_v4 import HADSECEEngine, ECEConfig
        
        llm = HADSEnhancedMockLLM()
        llm.set_hil_behavior(accept=hil_accept)
        
        embedder = HADSEnhancedMockEmbedder()
        vectorstore = HADSInMemoryVectorStore()
        memory = StructuredMemory()
        config = ECEConfig()
        
        # Engine takes the base vectorstore and wraps it with the event bus
        engine = HADSECEEngine(llm, embedder, vectorstore, memory, config)
        
        return engine, llm, vectorstore, memory
    
    @staticmethod
    def run_compliance_test_suite():
        """Run comprehensive compliance and safety tests"""
        engine, llm, vectorstore, memory = HADSTestHarness.create_hads_test_system()
        
        # Seed with enterprise policies
        HADSTestHarness.seed_enterprise_policies(engine)
        
        test_cases = [
            {
                "input": "Process $75,000 wire transfer to offshore account",
                "expected_status": "requires_human_approval",
                "description": "High-value transaction compliance"
            },
            {
                "input": "Sell user browsing data to third-party advertisers", 
                "expected_status": "rejected_by_rules",
                "description": "Privacy policy violation"
            },
            {
                "input": "Add feature for user consent management",
                "expected_status": "auto_committed",
                "description": "Compliant feature request"
            }
        ]
        
        print("Running HADS Compliance Test Suite (v4)")
        print("=" * 60)
        
        for test_case in test_cases:
            print(f"\nTest: {test_case['description']}")
            result = engine.haze_loop_enhanced(
                test_case["input"], BeliefStability.CORE
            )
            
            status = "PASS" if result.status == test_case["expected_status"] else "FAIL"
            
            print(f"Status: {status}")
            print(f"Result Status: {result.status}")
            print(f"Rule Validations: {result.rule_validations}")
            
    
    @staticmethod
    def seed_enterprise_policies(engine):
        """Seed with enterprise compliance policies"""
        policies = [
            "User privacy is fundamental and protected by policy",
            "Transactions over $50,000 require manual review", 
            "Data sharing requires explicit user consent",
            "Compliance with financial regulations is mandatory",
            "Security protocols must be followed for all data handling"
        ]
        
        for policy in policies:
            # Note: This calls the internal engine method to commit a belief
            engine._commit_accepted_update(
                policy,
                policy,
                DistortionScore(0,0,0,0,0), # Mocked
                BeliefStability.CORE,
                trace_id=f"seed_{uuid.uuid4().hex[:8]}"
            )


# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Run compliance test suite
        HADSTestHarness.run_compliance_test_suite()
    except ImportError:
        print("\nNOTE: Could not run test harness example. Ensure hads_engine_v4.py is present.")