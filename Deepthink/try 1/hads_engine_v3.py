# eidetic_continuum/hads_engine_v3.py
"""
Enhanced HADS (Hybrid Agentic Decision System) Engine
Integrating deterministic rule engines with probabilistic LLM reasoning
for auditable, scalable enterprise AI systems.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import math
import uuid
from collections import defaultdict
from pydantic import BaseModel, Field, validator
import threading
from concurrent.futures import ThreadPoolExecutor


# ==================== HADS CORE MODELS ====================

class HADSTraceRecord(BaseModel):
    """Unified cross-layer decision tracing"""
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")
    timestamp: float = Field(default_factory=time.time)
    
    # Deterministic layer data
    rule_firings: List[str] = Field(default_factory=list)
    distortion_scores: Optional[Dict[str, float]] = None
    stability_level: str = "working"
    
    # Probabilistic layer data
    llm_reasoning_steps: List[str] = Field(default_factory=list)
    agent_actions: List[str] = Field(default_factory=list)
    
    # Cross-layer correlation
    layer_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    final_decision_reconciliation: str = ""
    audit_verification_hash: str = ""

    class Config:
        extra = "forbid"  # Strict schema enforcement

    @validator('audit_verification_hash', always=True)
    def compute_verification_hash(cls, v, values):
        """Compute cryptographic hash for audit integrity"""
        import hashlib
        data = f"{values.get('trace_id', '')}{values.get('timestamp', '')}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class HADSDeterministicRule(BaseModel):
    """Structured rule definition for deterministic core"""
    rule_id: str
    condition: str
    action: str
    priority: int = 1
    stability_requirement: str = "working"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HADSExecutionResult(BaseModel):
    """Structured result from HADS execution"""
    trace_id: str
    status: str
    interpretation: str
    confidence: float
    distortion_metrics: Dict[str, float]
    rule_validations: List[str]
    layer_correlation: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)


# ==================== ENHANCED CORE ENGINE ====================

def cosine_sim(a, b):
    """Enhanced with error handling"""
    try:
        na = math.sqrt(sum(x * x for x in a)) + 1e-12
        nb = math.sqrt(sum(x * x for x in b)) + 1e-12
        return sum(x * y for x, y in zip(a, b)) / (na * nb)
    except:
        return 0.0


class BeliefStability(Enum):
    CORE = "core"
    WORKING = "working" 
    EPHEMERAL = "ephemeral"


class HILAction(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    DEFER = "defer"


@dataclass
class HILResponse:
    action: HILAction
    modified_delta: Optional[str] = None
    confidence: float = 1.0
    notes: str = ""
    trace_id: str = ""


@dataclass
class DistortionScore:
    semantic_distance: float
    contradiction_score: float
    temporal_relevance: float
    cluster_coherence: float
    uncertainty: float

    @property
    def composite(self) -> float:
        return (
            0.35 * self.semantic_distance +
            0.35 * self.contradiction_score +
            0.15 * (1.0 - self.temporal_relevance) +
            0.15 * (1.0 - self.cluster_coherence)
        )


@dataclass
class BeliefNode:
    id: str
    text: str
    embedding: List[float]
    stability: BeliefStability
    confidence: float
    timestamp: float
    sources: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    version: int = 1
    parent_version: Optional[str] = None
    trace_id: str = ""  # HADS enhancement


@dataclass
class ProvenanceRecord:
    belief_id: str
    source_chain: List[str]
    critiques: List[str]
    evidence_docs: List[str]
    trust_score: float
    human_approved: bool
    timestamp: float
    trace_id: str = ""  # HADS enhancement
    rule_validations: List[str] = field(default_factory=list)  # HADS enhancement


class HADSDeterministicCore:
    """
    Deterministic rule engine simulating C-core Rete algorithm
    Provides 100% auditable, consistent decision making
    """
    
    def __init__(self):
        self.rules: Dict[str, HADSDeterministicRule] = {}
        self.rule_graph: Dict[str, List[str]] = {}
        self.execution_log: List[Dict] = []
        self.lock = threading.RLock()
        
        # Load default enterprise rules
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Pre-load enterprise compliance and safety rules"""
        default_rules = [
            HADSDeterministicRule(
                rule_id="compliance_high_value",
                condition="transaction_amount > 50000",
                action="REQUIRE_MANUAL_REVIEW",
                priority=10,
                stability_requirement="core"
            ),
            HADSDeterministicRule(
                rule_id="privacy_fundamental", 
                condition="'user_data' in context and 'sell' in proposal",
                action="REJECT",
                priority=10,
                stability_requirement="core"
            ),
            HADSDeterministicRule(
                rule_id="safety_guardrail",
                condition="contradiction_score > 0.8",
                action="REQUIRE_HUMAN_APPROVAL", 
                priority=8,
                stability_requirement="working"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: HADSDeterministicRule):
        """Add rule to deterministic core"""
        with self.lock:
            self.rules[rule.rule_id] = rule
    
    def execute_rules(self, facts: Dict[str, Any], trace_id: str) -> Tuple[List[str], List[str]]:
        """
        Execute rules against facts with audit trail
        Returns: (actions, audit_trail)
        """
        with self.lock:
            actions = []
            audit_trail = []
            
            # Sort rules by priority
            sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority, reverse=True)
            
            for rule in sorted_rules:
                try:
                    # Simple condition evaluation (in production: use proper rule engine)
                    condition_met = self._evaluate_condition(rule.condition, facts)
                    
                    audit_trail.append(f"Rule {rule.rule_id}: condition={condition_met}")
                    
                    if condition_met:
                        actions.append(rule.action)
                        audit_trail.append(f"Action triggered: {rule.action}")
                        
                except Exception as e:
                    audit_trail.append(f"Rule {rule.rule_id} error: {str(e)}")
            
            # Log execution for audit
            self.execution_log.append({
                "trace_id": trace_id,
                "timestamp": time.time(),
                "facts": facts,
                "actions": actions,
                "audit_trail": audit_trail
            })
            
            return actions, audit_trail
    
    def _evaluate_condition(self, condition: str, facts: Dict[str, Any]) -> bool:
        """Simplified condition evaluation"""
        # In production: integrate with Durable Rules engine
        try:
            # Simple keyword-based evaluation
            if ">" in condition:
                parts = condition.split(">")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())
                    return facts.get(key, 0) > value
            
            if "in context" in condition and "in proposal" in condition:
                # Complex condition evaluation
                context_ok = any(word in str(facts.get("context", "")).lower() 
                               for word in ["user_data", "personal"])
                proposal_ok = any(word in str(facts.get("proposal", "")).lower()
                                for word in ["sell", "share", "distribute"])
                return context_ok and proposal_ok
            
            # Default: keyword presence check
            return any(keyword in condition.lower() 
                      for keyword in ["contradiction", "high_risk", "compliance"])
                      
        except Exception:
            return False


class HADSEventBus:
    """
    Event-driven architecture foundation
    Simulates Kafka-like message bus for loose coupling
    """
    
    def __init__(self):
        self.topics: Dict[str, List[callable]] = defaultdict(list)
        self.message_log: List[Dict] = []
        self.trace_correlations: Dict[str, List[str]] = {}
    
    def subscribe(self, topic: str, callback: callable):
        """Subscribe callback to topic"""
        self.topics[topic].append(callback)
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish message to topic with trace correlation"""
        trace_id = message.get("trace_id", "unknown")
        
        # Log message
        self.message_log.append({
            "timestamp": time.time(),
            "topic": topic,
            "message": message,
            "trace_id": trace_id
        })
        
        # Track trace correlation
        if trace_id not in self.trace_correlations:
            self.trace_correlations[trace_id] = []
        self.trace_correlations[trace_id].append(topic)
        
        # Notify subscribers
        for callback in self.topics[topic]:
            try:
                callback(message)
            except Exception as e:
                print(f"Event bus callback error: {e}")
    
    def get_trace_events(self, trace_id: str) -> List[Dict]:
        """Get all events for a trace ID"""
        return [msg for msg in self.message_log if msg.get("trace_id") == trace_id]


class EnhancedVectorStore:
    """Extended with HADS tracing and event integration"""
    
    def __init__(self, base_store, event_bus: HADSEventBus):
        self.base_store = base_store
        self.doc_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.event_bus = event_bus
    
    def add_document(self, id: str, text: str, embedding: List[float], 
                    metadata: Dict[str, Any], trace_id: str = ""):
        """Enhanced with event publishing"""
        self.base_store.add_document(id, text, embedding, metadata)
        
        # Publish document addition event
        self.event_bus.publish("vectorstore.document_added", {
            "doc_id": id,
            "text_preview": text[:100],
            "trace_id": trace_id,
            "timestamp": time.time()
        })
    
    def query_by_embedding(self, embedding: List[float], k: int = 5, 
                          rerank: bool = True, trace_id: str = "") -> List[Dict[str, Any]]:
        """Enhanced with tracing and diversity reranking"""
        results = self.base_store.query_by_embedding(
            embedding, k=k * 2 if rerank else k
        )
        
        # Publish query event
        self.event_bus.publish("vectorstore.query_executed", {
            "results_count": len(results),
            "requested_k": k,
            "trace_id": trace_id,
            "timestamp": time.time()
        })
        
        if rerank and len(results) > k:
            # MMR-like diversity reranking
            selected = []
            while len(selected) < k and results:
                if not selected:
                    selected.append(results.pop(0))
                else:
                    best_idx = 0
                    best_score = -1
                    for i, result in enumerate(results):
                        min_sim = min(
                            cosine_sim(result["embedding"], s["embedding"])
                            for s in selected
                        )
                        if min_sim > best_score:
                            best_score = min_sim
                            best_idx = i
                    selected.append(results.pop(best_idx))
            return selected
        
        return results[:k]


class HADSECEEngine:
    """
    Enhanced ECE Engine with HADS architecture
    Integrates deterministic core with probabilistic reasoning
    """
    
    def __init__(self, llm, embedder, vectorstore, memory, config=None):
        self.llm = llm
        self.embedder = embedder
        self.event_bus = HADSEventBus()
        self.vectorstore = EnhancedVectorStore(vectorstore, self.event_bus)
        self.memory = memory
        self.config = config or ECEConfig()
        
        # HADS components
        self.deterministic_core = HADSDeterministicCore()
        self.trace_records: Dict[str, HADSTraceRecord] = {}
        
        # Setup event subscriptions
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event-driven architecture handlers"""
        self.event_bus.subscribe("deterministic.rules_executed", 
                               self._handle_rules_executed)
        self.event_bus.subscribe("llm.interpretation_generated",
                               self._handle_llm_interpretation)
    
    def _handle_rules_executed(self, message: Dict):
        """Handle rule execution events"""
        trace_id = message.get("trace_id")
        if trace_id in self.trace_records:
            self.trace_records[trace_id].rule_firings.extend(
                message.get("actions", [])
            )
    
    def _handle_llm_interpretation(self, message: Dict):
        """Handle LLM interpretation events"""
        trace_id = message.get("trace_id")
        if trace_id in self.trace_records:
            self.trace_records[trace_id].llm_reasoning_steps.append(
                message.get("interpretation", "")
            )
    
    def create_trace_record(self, stability: BeliefStability) -> HADSTraceRecord:
        """Create new trace record for cross-layer tracking"""
        trace_record = HADSTraceRecord(stability_level=stability.value)
        self.trace_records[trace_record.trace_id] = trace_record
        return trace_record
    
    def haze_loop_enhanced(self, input_text: str, 
                          stability: BeliefStability = BeliefStability.WORKING) -> HADSExecutionResult:
        """
        Enhanced haze_loop with HADS architecture
        """
        # 1) Create unified trace
        trace_record = self.create_trace_record(stability)
        
        # 2) Deterministic pre-processing (Clear the Haze)
        rule_facts = self._extract_rule_facts(input_text, stability)
        rule_actions, rule_audit = self.deterministic_core.execute_rules(
            rule_facts, trace_record.trace_id
        )
        
        # Publish rule execution
        self.event_bus.publish("deterministic.rules_executed", {
            "trace_id": trace_record.trace_id,
            "actions": rule_actions,
            "audit_trail": rule_audit,
            "facts": rule_facts
        })
        
        # 3) Context retrieval with tracing
        context_summary = self.memory.get_summary()
        emb_input = self.embedder.embed(input_text)
        docs = self.vectorstore.query_by_embedding(
            emb_input, k=self.config.k, rerank=True, 
            trace_id=trace_record.trace_id
        )
        
        # 4) LLM reasoning with constraints (Achieve Infinity)
        llm_constraints = self._convert_rules_to_constraints(rule_actions)
        interpretation_result = self._llm_reason_with_constraints(
            input_text, context_summary, llm_constraints, trace_record.trace_id
        )
        
        # 5) Cross-layer decision reconciliation
        final_decision = self._reconcile_layers(
            trace_record.trace_id, rule_actions, interpretation_result, docs
        )
        
        # 6) Build comprehensive result
        return HADSExecutionResult(
            trace_id=trace_record.trace_id,
            status=final_decision["status"],
            interpretation=final_decision["interpretation"],
            confidence=final_decision.get("confidence", 0.5),
            distortion_metrics=final_decision.get("distortion_metrics", {}),
            rule_validations=rule_audit,
            layer_correlation={
                "deterministic_actions": rule_actions,
                "probabilistic_reasoning": interpretation_result.get("reasoning_steps", []),
                "reconciliation_strategy": final_decision.get("reconciliation_strategy", "")
            }
        )
    
    def _extract_rule_facts(self, input_text: str, stability: BeliefStability) -> Dict[str, Any]:
        """Extract facts for rule engine from input"""
        return {
            "input_text": input_text,
            "stability_level": stability.value,
            "timestamp": time.time(),
            "context": self.memory.get_summary(include_ephemeral=False),
            "proposal": input_text  # For compliance rule checking
        }
    
    def _convert_rules_to_constraints(self, rule_actions: List[str]) -> Dict[str, Any]:
        """Convert rule actions to LLM constraints"""
        constraints = {
            "must_avoid": [],
            "must_include": [],
            "validation_requirements": []
        }
        
        for action in rule_actions:
            if "REJECT" in action:
                constraints["must_avoid"].append("violate_compliance_policies")
            if "REQUIRE_MANUAL_REVIEW" in action:
                constraints["validation_requirements"].append("human_approval_required")
            if "REQUIRE_HUMAN_APPROVAL" in action:
                constraints["validation_requirements"].append("high_risk_approval_needed")
        
        return constraints
    
    def _llm_reason_with_constraints(self, input_text: str, context: str, 
                                   constraints: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
        """LLM reasoning with deterministic constraints"""
        # Enhanced prompt with constraints
        prompt = {
            "role": "hads_agent",
            "input": input_text,
            "context": context,
            "constraints": constraints,
            "trace_id": trace_id,
            "instruction": "Reason within provided constraints and document reasoning steps"
        }
        
        raw_response = self.llm.generate(json.dumps(prompt))
        
        # Parse and track reasoning
        interpretation = self._parse_llm_response(raw_response)
        
        # Publish interpretation event
        self.event_bus.publish("llm.interpretation_generated", {
            "trace_id": trace_id,
            "interpretation": interpretation.get("interpretation", ""),
            "confidence": interpretation.get("confidence", 0.5),
            "reasoning_steps": interpretation.get("reasoning_steps", [])
        })
        
        return interpretation
    
    def _parse_llm_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse LLM response with enhanced structure"""
        try:
            response = json.loads(raw_response)
            return {
                "interpretation": response.get("interpretation", raw_response),
                "confidence": response.get("confidence", 0.5),
                "reasoning_steps": response.get("reasoning_steps", []),
                "constraints_applied": response.get("constraints_applied", [])
            }
        except:
            return {
                "interpretation": raw_response,
                "confidence": 0.5,
                "reasoning_steps": [raw_response[:200]],
                "constraints_applied": []
            }
    
    def _reconcile_layers(self, trace_id: str, rule_actions: List[str], 
                         interpretation: Dict[str, Any], docs: List[Dict]) -> Dict[str, Any]:
        """Reconcile deterministic and probabilistic layers"""
        # Measure distortion
        distortion = self.measure_distortion_enhanced(
            interpretation["interpretation"], docs, 
            BeliefStability(self.trace_records[trace_id].stability_level)
        )
        
        # Apply rule-based overrides
        final_decision = interpretation.copy()
        
        # If rules require rejection, override LLM
        if any("REJECT" in action for action in rule_actions):
            final_decision.update({
                "status": "rejected_by_rules",
                "interpretation": f"REJECTED: {interpretation['interpretation']}",
                "confidence": 0.1,
                "reconciliation_strategy": "rule_override"
            })
        # If rules require human approval, mark for review
        elif any("REQUIRE_HUMAN_APPROVAL" in action for action in rule_actions):
            final_decision.update({
                "status": "requires_human_approval",
                "reconciliation_strategy": "human_escalation"
            })
        else:
            # Use LLM decision with rule validation
            final_decision.update({
                "status": "auto_committed",
                "reconciliation_strategy": "validated_acceptance",
                "distortion_metrics": {
                    "composite": distortion.composite,
                    "semantic_distance": distortion.semantic_distance,
                    "contradiction_score": distortion.contradiction_score
                }
            })
        
        return final_decision
    
    def measure_distortion_enhanced(self, interpretation_text: str, 
                                  context_docs: List[Dict[str, Any]],
                                  stability: BeliefStability) -> DistortionScore:
        """Enhanced distortion measurement with HADS tracing"""
        # Implementation from original ECE engine
        e_i = self.embedder.embed(interpretation_text)
        
        # Semantic distance
        sims = []
        weighted_sims = []
        current_time = time.time()
        
        for d in context_docs:
            e_d = d.get("embedding")
            if e_d is None:
                e_d = self.embedder.embed(d.get("text", ""))
            
            sim = cosine_sim(e_i, e_d)
            sims.append(sim)
            
            # Apply temporal decay
            doc_time = d.get("metadata", {}).get("time", current_time)
            age_hours = (current_time - doc_time) / 3600
            decay = self.config.temporal_decay_factor ** age_hours
            weighted_sims.append(sim * decay)
        
        max_sim = max(sims) if sims else 0.0
        semantic_distance = 1.0 - max_sim
        
        # Temporal relevance
        temporal_relevance = max(weighted_sims) if weighted_sims else 0.0
        
        # Contradiction detection
        contradiction_score = 0.0
        if hasattr(self.memory, "belief_graph"):
            conflicts = self.memory.belief_graph.find_conflicts(interpretation_text, e_i)
            if conflicts:
                contradiction_score = max(score for _, score in conflicts)
        
        # Cluster coherence
        cluster_coherence = 1.0
        if len(sims) > 1:
            mean_sim = sum(sims) / len(sims)
            variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
            cluster_coherence = 1.0 - min(variance * 2, 1.0)
        
        uncertainty = 1.0 - cluster_coherence
        
        return DistortionScore(
            semantic_distance=semantic_distance,
            contradiction_score=contradiction_score,
            temporal_relevance=temporal_relevance,
            cluster_coherence=cluster_coherence,
            uncertainty=uncertainty
        )
    
    def get_trace_audit_report(self, trace_id: str) -> Dict[str, Any]:
        """Generate comprehensive audit report for trace"""
        if trace_id not in self.trace_records:
            return {"error": "Trace not found"}
        
        trace = self.trace_records[trace_id]
        events = self.event_bus.get_trace_events(trace_id)
        
        return {
            "trace_id": trace_id,
            "timestamp": trace.timestamp,
            "stability_level": trace.stability_level,
            "deterministic_actions": trace.rule_firings,
            "probabilistic_reasoning": trace.llm_reasoning_steps,
            "layer_interactions": trace.layer_interactions,
            "event_flow": events,
            "audit_verification": trace.audit_verification_hash
        }


# ==================== CONFIGURATION ====================

class ECEConfig:
    def __init__(
        self,
        k: int = 5,
        tau_refine: float = 0.35,
        tau_accept: float = 0.20,
        tau_hil: float = 0.60,
        max_refine_iters: int = 3,
        convergence_epsilon: float = 0.02,
        temporal_decay_factor: float = 0.95,
        enable_rollback: bool = True,
        max_history_depth: int = 100,
    ):
        self.k = k
        self.tau_refine = tau_refine
        self.tau_accept = tau_accept
        self.tau_hil = tau_hil
        self.max_refine_iters = max_refine_iters
        self.convergence_epsilon = convergence_epsilon
        self.temporal_decay_factor = temporal_decay_factor
        self.enable_rollback = enable_rollback
        self.max_history_depth = max_history_depth


# ==================== COMPATIBILITY WRAPPERS ====================

class StructuredMemory:
    """Compatibility wrapper for original memory structure"""
    def __init__(self):
        self.core_beliefs: List[str] = []
        self.working_knowledge: List[str] = []
        self.ephemeral_context: List[str] = []
        self.belief_graph = BeliefGraph()
    
    def get_summary(self, include_ephemeral: bool = True) -> str:
        parts = []
        if self.core_beliefs:
            parts.append("CORE BELIEFS:\n" + "\n".join(self.core_beliefs[-5:]))
        if self.working_knowledge:
            parts.append("WORKING KNOWLEDGE:\n" + "\n".join(self.working_knowledge[-10:]))
        if include_ephemeral and self.ephemeral_context:
            parts.append("RECENT CONTEXT:\n" + "\n".join(self.ephemeral_context[-5:]))
        return "\n\n".join(parts)
    
    def update_with(self, delta_text: str, source_meta: Dict[str, Any]):
        stability = source_meta.get("stability", BeliefStability.WORKING)
        if stability == BeliefStability.CORE:
            self.core_beliefs.append(delta_text)
        elif stability == BeliefStability.WORKING:
            self.working_knowledge.append(delta_text)
        else:
            self.ephemeral_context.append(delta_text)
            if len(self.ephemeral_context) > 20:
                self.ephemeral_context.pop(0)


class BeliefGraph:
    """Compatibility wrapper"""
    def __init__(self):
        self.nodes: Dict[str, BeliefNode] = {}
        self.edges: Dict[Tuple[str, str], str] = {}
        self.contradiction_log: List[Dict[str, Any]] = []
    
    def find_conflicts(self, new_text: str, new_embedding: List[float]) -> List[Tuple[str, float]]:
        conflicts = []
        for node_id, node in self.nodes.items():
            if node.stability == BeliefStability.CORE:
                sim = cosine_sim(new_embedding, node.embedding)
                if sim < 0.3:
                    conflicts.append((node_id, 1.0 - sim))
        return conflicts


# ==================== USAGE EXAMPLE ====================

def demo_hads_engine():
    """Demonstrate the enhanced HADS engine"""
    from agents_v2 import EnhancedMockLLM, EnhancedMockEmbedder, InMemoryVectorStore
    
    # Create components
    llm = EnhancedMockLLM()
    embedder = EnhancedMockEmbedder()
    vectorstore = InMemoryVectorStore()
    memory = StructuredMemory()
    config = ECEConfig()
    
    # Create HADS engine
    engine = HADSECEEngine(llm, embedder, vectorstore, memory, config)
    
    # Test cases
    test_cases = [
        ("Add user analytics with consent", BeliefStability.WORKING),
        ("Sell user data to advertisers", BeliefStability.CORE),
        ("Process $75,000 transaction", BeliefStability.WORKING),
    ]
    
    for input_text, stability in test_cases:
        print(f"\n{'='*60}")
        print(f"INPUT: {input_text}")
        print(f"STABILITY: {stability.value}")
        print(f"{'='*60}")
        
        result = engine.haze_loop_enhanced(input_text, stability)
        
        print(f"STATUS: {result.status}")
        print(f"INTERPRETATION: {result.interpretation}")
        print(f"CONFIDENCE: {result.confidence:.3f}")
        print(f"RULE VALIDATIONS: {result.rule_validations}")
        
        # Show audit report
        audit = engine.get_trace_audit_report(result.trace_id)
        print(f"AUDIT EVENTS: {len(audit.get('event_flow', []))}")
        print(f"DETERMINISTIC ACTIONS: {audit.get('deterministic_actions', [])}")


if __name__ == "__main__":
    demo_hads_engine()