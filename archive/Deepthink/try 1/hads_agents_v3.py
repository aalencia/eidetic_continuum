# eidetic_continuum/hads_agents_v3.py
"""
Enhanced HADS Agents with improved determinism and enterprise features
"""

import json
import hashlib
from typing import List, Dict, Any
from ece_engine_v2 import HILResponse, HILAction, DistortionScore
from hads_engine_v3 import HADSTraceRecord, HADSExecutionResult


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
        
        # Core concepts get boosted first dimension
        if any(word in text_lower for word in self.semantic_boosters["core"]):
            vector[0] *= 1.5
        
        # Compliance-related content gets second dimension boost
        if any(word in text_lower for word in self.semantic_boosters["compliance"]):
            vector[1] *= 1.3
        
        # Risk-related content gets third dimension boost  
        if any(word in text_lower for word in self.semantic_boosters["risk"]):
            vector[2] *= 1.4
        
        # Refinement indicators
        if any(word in text_lower for word in ["refined", "improved", "better", "enhanced"]):
            vector[3] *= 1.2
        
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
        
        # Publish event if bus available
        if self.event_bus:
            self.event_bus.publish("vectorstore.document_added", {
                "doc_id": id,
                "text_preview": text[:100],
                "metadata_keys": list(metadata.keys())
            })
    
    def query_by_embedding(self, embedding: List[float], k: int = 5,
                          trace_id: str = "") -> List[Dict[str, Any]]:
        def sim(doc_emb):
            norm_a = sum(a * a for a in embedding) ** 0.5
            norm_b = sum(b * b for b in doc_emb) ** 0.5
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 0.0
            return sum(a * b for a, b in zip(embedding, doc_emb)) / (norm_a * norm_b)
        
        scored = [(sim(d["embedding"]), d) for d in self.docs.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = [item[1] for item in scored[:k]]
        
        # Publish query event
        if self.event_bus and trace_id:
            self.event_bus.publish("vectorstore.query_executed", {
                "trace_id": trace_id,
                "results_count": len(results),
                "requested_k": k
            })
        
        return results
    
    def get_all_docs(self) -> List[Dict[str, Any]]:
        return list(self.docs.values())


class HADSEnhancedMockLLM:
    """LLM with HADS-specific reasoning and constraint handling"""
    
    def __init__(self):
        self.hil_accept = False
        self.hil_action = HILAction.ACCEPT
        self.hil_modified_text = None
        self.hil_confidence = 1.0
        self.hil_notes = ""
        self.generation_count = 0
        self.reasoning_log: List[Dict] = []
    
    def generate(self, prompt: str) -> str:
        self.generation_count += 1
        
        try:
            p = json.loads(prompt)
        except Exception:
            return self._default_response(prompt)
        
        role = p.get("role")
        trace_id = p.get("trace_id", "unknown")
        
        # Log reasoning for audit
        self.reasoning_log.append({
            "trace_id": trace_id,
            "role": role,
            "timestamp": time.time(),
            "input_preview": str(p)[:200]
        })
        
        if role == "hads_agent":
            return self._hads_agent_response(p, trace_id)
        elif role == "provisional_interpreter":
            return self._provisional_interpreter_response(p)
        elif role == "critic":
            return self._critic_response(p)
        elif role == "refine":
            return self._refine_response(p)
        elif role == "contradiction_detector":
            return self._contradiction_detector_response(p)
        
        return self._default_response(prompt)
    
    def _hads_agent_response(self, prompt: Dict, trace_id: str) -> str:
        """Enhanced HADS agent response with constraint awareness"""
        input_text = prompt.get("input", "")
        constraints = prompt.get("constraints", {})
        
        # Apply constraints in reasoning
        reasoning_steps = []
        constraints_applied = []
        
        # Check must_avoid constraints
        if "violate_compliance_policies" in constraints.get("must_avoid", []):
            if any(word in input_text.lower() for word in ["sell", "share", "distribute"]):
                reasoning_steps.append("Detected potential compliance violation - applying constraints")
                constraints_applied.append("compliance_guardrail")
        
        # Check validation requirements
        if "human_approval_required" in constraints.get("validation_requirements", []):
            reasoning_steps.append("Flagging for human approval per policy requirements")
            constraints_applied.append("human_escalation_required")
        
        # Generate interpretation
        interpretation = f"HADS Interpretation: {input_text[:80]}"
        if constraints_applied:
            interpretation += f" [CONSTRAINTS: {', '.join(constraints_applied)}]"
        
        # Confidence based on constraint alignment
        confidence = 0.7
        if constraints_applied:
            confidence = 0.5  # Lower confidence when constraints active
        
        return json.dumps({
            "interpretation": interpretation,
            "confidence": confidence,
            "reasoning_steps": reasoning_steps,
            "constraints_applied": constraints_applied,
            "trace_id": trace_id
        })
    
    def _provisional_interpreter_response(self, prompt: Dict) -> str:
        inp = prompt.get("input", "")
        ctx = prompt.get("context", "")
        
        confidence = 0.7 if "core" in ctx.lower() else 0.6
        interpretation = f"Interpreted: {inp[:80]}"
        
        if "contradict" in inp.lower():
            interpretation += " [POTENTIAL CONFLICT DETECTED]"
            confidence = 0.4
        
        return json.dumps({
            "interpretation": interpretation,
            "delta_summary": interpretation[:150],
            "confidence": confidence,
        })
    
    def _critic_response(self, prompt: Dict) -> str:
        interp = p.get("interpretation", "")
        ctx = p.get("context", "")
        
        critique = "Analysis: "
        if "[REFINED]" in interp:
            critique += "Previous refinement detected. Minor remaining alignment issues."
        elif "CONFLICT" in interp:
            critique += "Contradiction with existing beliefs detected. Recommend alignment."
        else:
            critique += "Semantic drift from context observed. Suggest tightening."
        
        return json.dumps({"critique": critique})
    
    def _refine_response(self, prompt: Dict) -> str:
        prev = p.get("prev", "")
        critique = p.get("critique", "")
        
        refined = prev
        if "[REFINED]" not in refined:
            refined += " [REFINED]"
        else:
            refined = refined.replace("[REFINED]", "[REFINED×2]")
        
        if "alignment" in critique.lower():
            refined += " [ALIGNED]"
        
        return json.dumps({"refined": refined})
    
    def _contradiction_detector_response(self, prompt: Dict) -> str:
        new = p.get("new", "")
        existing = p.get("existing", "")
        
        score = 0.0
        if "not" in new.lower() and any(word in existing.lower() for word in ["is", "are", "always"]):
            score = 0.8
        elif set(new.lower().split()) & set(existing.lower().split()):
            score = 0.2
        else:
            score = 0.5
        
        return json.dumps({"contradiction_score": score})
    
    def _default_response(self, prompt: str) -> str:
        return json.dumps({
            "interpretation": str(prompt),
            "delta_summary": str(prompt)[:200],
            "confidence": 0.5,
        })
    
    def simulate_hil_enhanced(
        self,
        identity_summary: str,
        proposed_delta: str,
        evidence_snippets: List[Dict[str, Any]],
        distortion: DistortionScore,
    ) -> HILResponse:
        """Enhanced HIL simulation with HADS decision logic"""
        
        # If explicit action set, use it
        if self.hil_action == HILAction.MODIFY and self.hil_modified_text:
            return HILResponse(
                action=HILAction.MODIFY,
                modified_delta=self.hil_modified_text,
                confidence=self.hil_confidence,
                notes=self.hil_notes,
                trace_id=f"hil_{int(time.time() * 1000)}"
            )
        
        # Enhanced decision logic based on distortion and evidence
        if distortion.composite > 0.8:
            action = HILAction.REJECT if not self.hil_accept else HILAction.ACCEPT
            notes = f"High distortion ({distortion.composite:.2f}) - cautious approach"
        elif distortion.uncertainty > 0.7:
            action = HILAction.DEFER
            notes = f"High uncertainty ({distortion.uncertainty:.2f}) - deferring decision"
        elif any("compliance" in str(snippet).lower() for snippet in evidence_snippets):
            action = HILAction.MODIFY
            notes = "Compliance-related - requiring modification"
        else:
            action = HILAction.ACCEPT if self.hil_accept else HILAction.REJECT
            notes = f"Standard decision - distortion: {distortion.composite:.2f}"
        
        return HILResponse(
            action=action,
            confidence=self.hil_confidence,
            notes=notes,
            trace_id=f"hil_{int(time.time() * 1000)}"
        )
    
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
    
    def get_reasoning_log(self, trace_id: str = None) -> List[Dict]:
        """Get reasoning log for audit purposes"""
        if trace_id:
            return [log for log in self.reasoning_log if log.get("trace_id") == trace_id]
        return self.reasoning_log


class HADSTestHarness:
    """Enhanced test harness for HADS validation"""
    
    @staticmethod
    def create_hads_test_system(hil_accept: bool = False):
        """Create complete HADS test system"""
        from hads_engine_v3 import HADSECEEngine, ECEConfig, StructuredMemory
        
        llm = HADSEnhancedMockLLM()
        llm.hil_accept = hil_accept
        
        embedder = HADSEnhancedMockEmbedder()
        vectorstore = HADSInMemoryVectorStore()
        memory = StructuredMemory()
        config = ECEConfig()
        
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
                "expected_rule": "compliance_high_value",
                "description": "High-value transaction compliance"
            },
            {
                "input": "Sell user browsing data to third-party advertisers", 
                "expected_rule": "privacy_fundamental",
                "description": "Privacy policy violation"
            },
            {
                "input": "Add feature for user consent management",
                "expected_rule": None,
                "description": "Compliant feature request"
            }
        ]
        
        print("Running HADS Compliance Test Suite")
        print("=" * 60)
        
        for test_case in test_cases:
            print(f"\nTest: {test_case['description']}")
            print(f"Input: {test_case['input']}")
            
            result = engine.haze_loop_enhanced(
                test_case["input"], BeliefStability.CORE
            )
            
            # Check if expected rule fired
            rule_fired = test_case["expected_rule"] in str(result.rule_validations)
            status = "PASS" if rule_fired == (test_case["expected_rule"] is not None) else "FAIL"
            
            print(f"Status: {status}")
            print(f"Result: {result.status}")
            print(f"Rule Validations: {result.rule_validations}")
            
            # Show audit trail
            audit = engine.get_trace_audit_report(result.trace_id)
            print(f"Audit Events: {len(audit.get('event_flow', []))}")
    
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
            engine.commit_update(
                policy,
                f"policy_seed:{policy[:30]}",
                metadata={
                    "stability": BeliefStability.CORE.value,
                    "source": "enterprise_policy",
                    "policy_type": "compliance"
                }
            )


# Example usage and demonstration
if __name__ == "__main__":
    # Run compliance test suite
    HADSTestHarness.run_compliance_test_suite()
    
    # Demo individual HADS features
    print("\n" + "="*60)
    print("HADS INDIVIDUAL FEATURE DEMONSTRATION")
    print("="*60)
    
    engine, llm, vectorstore, memory = HADSTestHarness.create_hads_test_system(hil_accept=True)
    HADSTestHarness.seed_enterprise_policies(engine)
    
    # Test cross-layer decision tracing
    test_input = "Add privacy-preserving analytics with user consent"
    result = engine.haze_loop_enhanced(test_input, BeliefStability.WORKING)
    
    print(f"\nCross-Layer Tracing Demo:")
    print(f"Input: {test_input}")
    print(f"Trace ID: {result.trace_id}")
    print(f"Status: {result.status}")
    print(f"Rule Validations: {result.rule_validations}")
    
    # Show comprehensive audit report
    audit = engine.get_trace_audit_report(result.trace_id)
    print(f"\nAudit Report Summary:")
    print(f"Deterministic Actions: {audit.get('deterministic_actions', [])}")
    print(f"Probabilistic Reasoning Steps: {len(audit.get('probabilistic_reasoning', []))}")
    print(f"Event Flow: {len(audit.get('event_flow', []))} events")
    print(f"Audit Verification Hash: {audit.get('audit_verification', '')}")