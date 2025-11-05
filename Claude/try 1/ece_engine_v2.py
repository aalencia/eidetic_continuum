# eidetic_continuum/ece_engine_v2.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import math
from collections import defaultdict


def cosine_sim(a, b):
    na = math.sqrt(sum(x * x for x in a)) + 1e-12
    nb = math.sqrt(sum(x * x for x in b)) + 1e-12
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class BeliefStability(Enum):
    CORE = "core"  # Foundational beliefs, highest threshold
    WORKING = "working"  # Regular knowledge, medium threshold
    EPHEMERAL = "ephemeral"  # Context, low threshold


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


@dataclass
class DistortionScore:
    semantic_distance: float
    contradiction_score: float
    temporal_relevance: float
    cluster_coherence: float
    uncertainty: float

    @property
    def composite(self) -> float:
        # Weighted composite distortion
        return (
            0.35 * self.semantic_distance
            + 0.35 * self.contradiction_score
            + 0.15 * (1.0 - self.temporal_relevance)
            + 0.15 * (1.0 - self.cluster_coherence)
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


@dataclass
class ProvenanceRecord:
    belief_id: str
    source_chain: List[str]
    critiques: List[str]
    evidence_docs: List[str]
    trust_score: float
    human_approved: bool
    timestamp: float


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


class BeliefGraph:
    def __init__(self):
        self.nodes: Dict[str, BeliefNode] = {}
        self.edges: Dict[Tuple[str, str], str] = {}  # (id1, id2) -> relationship
        self.contradiction_log: List[Dict[str, Any]] = []

    def add_node(self, node: BeliefNode):
        self.nodes[node.id] = node

    def add_edge(self, id1: str, id2: str, relationship: str):
        self.edges[(id1, id2)] = relationship

    def find_conflicts(
        self, new_text: str, new_embedding: List[float]
    ) -> List[Tuple[str, float]]:
        """Find potentially conflicting beliefs"""
        conflicts = []
        for node_id, node in self.nodes.items():
            if node.stability == BeliefStability.CORE:
                sim = cosine_sim(new_embedding, node.embedding)
                # Low similarity on core beliefs suggests potential conflict
                if sim < 0.3:
                    conflicts.append((node_id, 1.0 - sim))
        return conflicts

    def get_by_stability(self, stability: BeliefStability) -> List[BeliefNode]:
        return [n for n in self.nodes.values() if n.stability == stability]

    def log_contradiction(
        self, belief_id: str, conflicting_ids: List[str], reason: str
    ):
        self.contradiction_log.append(
            {
                "belief_id": belief_id,
                "conflicts_with": conflicting_ids,
                "reason": reason,
                "timestamp": time.time(),
            }
        )


class StructuredMemory:
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
            parts.append(
                "WORKING KNOWLEDGE:\n" + "\n".join(self.working_knowledge[-10:])
            )
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
            # Keep ephemeral context bounded
            if len(self.ephemeral_context) > 20:
                self.ephemeral_context.pop(0)


class EnhancedVectorStore:
    """Extended interface for chunked document storage"""

    def __init__(self, base_store):
        self.base_store = base_store
        self.doc_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_document(
        self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any]
    ):
        self.base_store.add_document(id, text, embedding, metadata)

    def add_chunked_document(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """Store document as multiple semantic chunks"""
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.base_store.add_document(
                chunk_id,
                chunk["text"],
                chunk["embedding"],
                {**chunk.get("metadata", {}), "parent_doc": doc_id},
            )
            self.doc_chunks[doc_id].append(chunk)

    def query_by_embedding(
        self, embedding: List[float], k: int = 5, rerank: bool = True
    ):
        results = self.base_store.query_by_embedding(
            embedding, k=k * 2 if rerank else k
        )
        if rerank:
            # Simple diversity reranking (MMR-like)
            selected = []
            while len(selected) < k and results:
                if not selected:
                    selected.append(results.pop(0))
                else:
                    # Pick result most dissimilar to already selected
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


class ECEEngine:
    def __init__(self, llm, embedder, vectorstore, memory, config: ECEConfig = None):
        self.llm = llm
        self.embedder = embedder
        self.vectorstore = EnhancedVectorStore(vectorstore)
        self.memory = (
            memory if isinstance(memory, StructuredMemory) else StructuredMemory()
        )
        self.config = config or ECEConfig()
        self.provenance: Dict[str, ProvenanceRecord] = {}
        self.version_history: List[Dict[str, Any]] = []

    def _provisional_prompt(self, input_text: str, context_summary: str) -> str:
        return json.dumps(
            {
                "role": "provisional_interpreter",
                "input": input_text,
                "context": context_summary,
                "instruction": "Provide interpretation with confidence score (0-1)",
            }
        )

    def _critic_prompt(self, interpretation: str, context_summary: str) -> str:
        return json.dumps(
            {
                "role": "critic",
                "interpretation": interpretation,
                "context": context_summary,
                "instruction": "Identify contradictions and provide specific critique",
            }
        )

    def _refine_prompt(self, prev: str, critique: str, context: str) -> str:
        return json.dumps(
            {
                "role": "refine",
                "prev": prev,
                "critique": critique,
                "context": context,
                "instruction": "Refine interpretation addressing critique",
            }
        )

    def _contradiction_check_prompt(self, new_text: str, existing_belief: str) -> str:
        return json.dumps(
            {
                "role": "contradiction_detector",
                "new": new_text,
                "existing": existing_belief,
                "instruction": "Score contradiction 0-1, where 1=direct conflict",
            }
        )

    def _parse_provisional(self, raw: str) -> Dict[str, Any]:
        try:
            j = json.loads(raw)
            return {
                "interpretation": j.get("interpretation", j.get("result", raw)),
                "delta_summary": j.get("delta_summary", j.get("delta", raw[:200])),
                "confidence": float(j.get("confidence", 0.5)),
            }
        except Exception:
            return {
                "interpretation": raw,
                "delta_summary": raw[:200],
                "confidence": 0.5,
            }

    def measure_distortion_enhanced(
        self,
        interpretation_text: str,
        context_docs: List[Dict[str, Any]],
        stability: BeliefStability = BeliefStability.WORKING,
    ) -> DistortionScore:
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
            decay = self.config.temporal_decay_factor**age_hours
            weighted_sims.append(sim * decay)

        max_sim = max(sims) if sims else 0.0
        semantic_distance = 1.0 - max_sim

        # Temporal relevance (how recent are similar docs)
        temporal_relevance = max(weighted_sims) if weighted_sims else 0.0

        # Contradiction detection (simplified - in production use NLI)
        contradiction_score = 0.0
        if hasattr(self.memory, "belief_graph"):
            conflicts = self.memory.belief_graph.find_conflicts(
                interpretation_text, e_i
            )
            if conflicts:
                contradiction_score = max(score for _, score in conflicts)

        # Cluster coherence (variance in similarities)
        cluster_coherence = 1.0
        if len(sims) > 1:
            mean_sim = sum(sims) / len(sims)
            variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
            cluster_coherence = 1.0 - min(variance * 2, 1.0)

        # Uncertainty based on confidence spread
        uncertainty = 1.0 - cluster_coherence

        return DistortionScore(
            semantic_distance=semantic_distance,
            contradiction_score=contradiction_score,
            temporal_relevance=temporal_relevance,
            cluster_coherence=cluster_coherence,
            uncertainty=uncertainty,
        )

    def commit_update(
        self,
        delta_text: str,
        input_text: str,
        metadata: Optional[dict] = None,
        provenance: Optional[ProvenanceRecord] = None,
    ):
        meta = metadata or {}
        meta.update({"time": time.time()})

        doc_id = f"patch_{int(time.time() * 1000)}"
        emb = self.embedder.embed(delta_text)

        self.vectorstore.add_document(doc_id, delta_text, emb, meta)

        # Update structured memory
        stability = meta.get("stability", BeliefStability.WORKING)
        self.memory.update_with(
            delta_text, {"origin_input": input_text, "stability": stability, **meta}
        )

        # Store provenance
        if provenance:
            self.provenance[doc_id] = provenance

        # Track version history for rollback
        if self.config.enable_rollback:
            self.version_history.append(
                {
                    "doc_id": doc_id,
                    "delta": delta_text,
                    "metadata": meta,
                    "timestamp": time.time(),
                }
            )
            if len(self.version_history) > self.config.max_history_depth:
                self.version_history.pop(0)

    def rollback(self, steps: int = 1) -> bool:
        """Rollback the last N commits"""
        if not self.config.enable_rollback or steps > len(self.version_history):
            return False

        for _ in range(steps):
            if self.version_history:
                self.version_history.pop()

        return True

    def human_in_loop_decision(
        self,
        identity_summary: str,
        proposed_delta: str,
        evidence_snippets: List[Dict[str, Any]],
        distortion: DistortionScore,
    ) -> HILResponse:
        """Enhanced HIL with multiple action types"""
        if hasattr(self.llm, "simulate_hil_enhanced"):
            return self.llm.simulate_hil_enhanced(
                identity_summary, proposed_delta, evidence_snippets, distortion
            )
        elif hasattr(self.llm, "simulate_hil"):
            accepted = self.llm.simulate_hil(
                identity_summary, proposed_delta, evidence_snippets
            )
            return HILResponse(
                action=HILAction.ACCEPT if accepted else HILAction.REJECT,
                confidence=1.0,
            )
        return HILResponse(action=HILAction.REJECT, confidence=0.0)

    def haze_loop(
        self, input_text: str, stability: BeliefStability = BeliefStability.WORKING
    ) -> Dict[str, Any]:
        """Holistic Adaptive Zero-distortion Engine - Enhanced"""

        # 1) Retrieve context with diversity
        context_summary = self.memory.get_summary()
        emb_input = self.embedder.embed(input_text)
        docs = self.vectorstore.query_by_embedding(
            emb_input, k=self.config.k, rerank=True
        )

        source_chain = []
        critiques = []

        # 2) Provisional interpret
        prompt = self._provisional_prompt(input_text, context_summary)
        source_chain.append("provisional_prompt")
        raw_out = self.llm.generate(prompt)
        prov = self._parse_provisional(raw_out)
        interpretation = prov["interpretation"]
        confidence = prov["confidence"]

        # 3) Enhanced distortion measurement
        distortion = self.measure_distortion_enhanced(interpretation, docs, stability)

        # Adjust thresholds based on stability level
        tau_hil = self.config.tau_hil
        tau_refine = self.config.tau_refine
        tau_accept = self.config.tau_accept

        if stability == BeliefStability.CORE:
            # Core beliefs require stricter thresholds
            tau_hil *= 0.7
            tau_refine *= 0.7
            tau_accept *= 0.7
        elif stability == BeliefStability.EPHEMERAL:
            # Ephemeral context is more permissive
            tau_hil *= 1.3
            tau_refine *= 1.3
            tau_accept *= 1.3

        # High uncertainty triggers refinement regardless of distortion
        if distortion.uncertainty > 0.7:
            tau_refine = min(tau_refine, distortion.composite - 0.05)

        # 4) Decision tree with enhanced logic
        if distortion.composite >= tau_hil:
            hil_response = self.human_in_loop_decision(
                context_summary, interpretation, docs[: self.config.k], distortion
            )

            if hil_response.action == HILAction.REJECT:
                return {
                    "status": "rejected_by_human",
                    "interpretation": interpretation,
                    "distortion": distortion,
                    "confidence": confidence,
                    "hil_notes": hil_response.notes,
                }
            elif hil_response.action == HILAction.MODIFY:
                interpretation = hil_response.modified_delta or interpretation
                confidence = hil_response.confidence
            elif hil_response.action == HILAction.DEFER:
                return {
                    "status": "deferred",
                    "interpretation": interpretation,
                    "distortion": distortion,
                    "confidence": confidence,
                }

            # ACCEPT or MODIFY case
            prov_record = ProvenanceRecord(
                belief_id=f"belief_{int(time.time() * 1000)}",
                source_chain=source_chain,
                critiques=critiques,
                evidence_docs=[d["id"] for d in docs[: self.config.k]],
                trust_score=confidence,
                human_approved=True,
                timestamp=time.time(),
            )

            self.commit_update(
                interpretation,
                input_text,
                metadata={
                    "distortion": distortion.composite,
                    "stability": stability.value,
                },
                provenance=prov_record,
            )

            return {
                "status": "committed_by_human",
                "interpretation": interpretation,
                "distortion": distortion,
                "confidence": confidence,
                "hil_action": hil_response.action.value,
            }

        # 5) Self-refinement with convergence detection
        if distortion.composite >= tau_refine:
            prev = interpretation
            prev_distortion = distortion.composite
            iters = 0

            while iters < self.config.max_refine_iters:
                crit_prompt = self._critic_prompt(prev, context_summary)
                critique = self.llm.generate(crit_prompt)
                critiques.append(critique)
                source_chain.append(f"refine_iter_{iters}")

                refine_prompt = self._refine_prompt(prev, critique, context_summary)
                refined_raw = self.llm.generate(refine_prompt)

                # Parse refined output
                prev = (
                    json.loads(refined_raw).get("refined")
                    if refined_raw.strip().startswith("{")
                    else refined_raw
                )

                # Measure new distortion
                distortion_new = self.measure_distortion_enhanced(prev, docs, stability)

                # Check convergence
                delta = abs(distortion_new.composite - prev_distortion)
                if delta < self.config.convergence_epsilon:
                    break

                # Backtrack if refinement made things worse
                if distortion_new.composite > prev_distortion * 1.2:
                    # Revert to previous iteration
                    break

                prev_distortion = distortion_new.composite
                iters += 1

            interpretation = prev
            distortion = self.measure_distortion_enhanced(
                interpretation, docs, stability
            )

        # 6) Final commit decision
        final_distortion = distortion.composite

        if final_distortion <= tau_accept:
            prov_record = ProvenanceRecord(
                belief_id=f"belief_{int(time.time() * 1000)}",
                source_chain=source_chain,
                critiques=critiques,
                evidence_docs=[d["id"] for d in docs[: self.config.k]],
                trust_score=confidence,
                human_approved=False,
                timestamp=time.time(),
            )

            self.commit_update(
                interpretation,
                input_text,
                metadata={"distortion": final_distortion, "stability": stability.value},
                provenance=prov_record,
            )

            return {
                "status": "auto_committed",
                "interpretation": interpretation,
                "distortion": distortion,
                "confidence": confidence,
                "refinement_iters": len(critiques),
            }
        else:
            # Final HIL check
            hil_response = self.human_in_loop_decision(
                context_summary, interpretation, docs[: self.config.k], distortion
            )

            if hil_response.action in [HILAction.ACCEPT, HILAction.MODIFY]:
                if hil_response.action == HILAction.MODIFY:
                    interpretation = hil_response.modified_delta or interpretation

                prov_record = ProvenanceRecord(
                    belief_id=f"belief_{int(time.time() * 1000)}",
                    source_chain=source_chain,
                    critiques=critiques,
                    evidence_docs=[d["id"] for d in docs[: self.config.k]],
                    trust_score=hil_response.confidence,
                    human_approved=True,
                    timestamp=time.time(),
                )

                self.commit_update(
                    interpretation,
                    input_text,
                    metadata={
                        "distortion": final_distortion,
                        "stability": stability.value,
                    },
                    provenance=prov_record,
                )

                return {
                    "status": "committed_by_human_post_refine",
                    "interpretation": interpretation,
                    "distortion": distortion,
                    "confidence": hil_response.confidence,
                    "refinement_iters": len(critiques),
                }

            return {
                "status": "rejected",
                "interpretation": interpretation,
                "distortion": distortion,
                "confidence": confidence,
                "hil_notes": hil_response.notes,
            }
