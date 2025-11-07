# eidetic_continuum/agents_v2.py
# Enhanced mock implementations with improved determinism and features

import json
import hashlib
from typing import List, Dict, Any
from ece_engine_v2 import HILResponse, HILAction, DistortionScore


class EnhancedMockEmbedder:
    """Deterministic embedder with better semantic simulation"""

    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        # Use hash for determinism but create varied embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Create vector from hash
        vector = []
        for i in range(self.dimensions):
            chunk = text_hash[i * 2 : (i + 1) * 2]
            val = int(chunk, 16) / 255.0  # Normalize to 0-1
            vector.append(val)

        # Add semantic features based on keywords
        if any(word in text.lower() for word in ["core", "fundamental", "essential"]):
            vector[0] *= 1.5  # Boost first dimension for core concepts

        if any(word in text.lower() for word in ["refined", "improved", "better"]):
            vector[1] *= 1.3  # Boost for refinement

        # Normalize
        norm = sum(x * x for x in vector) ** 0.5
        return [x / (norm + 1e-12) for x in vector]


class InMemoryVectorStore:
    """Enhanced with better retrieval"""

    def __init__(self):
        self.docs: Dict[str, Dict[str, Any]] = {}

    def add_document(
        self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any]
    ):
        self.docs[id] = {
            "id": id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

    def query_by_embedding(
        self, embedding: List[float], k: int = 5
    ) -> List[Dict[str, Any]]:
        def sim(doc_emb):
            norm_a = sum(a * a for a in embedding) ** 0.5
            norm_b = sum(b * b for b in doc_emb) ** 0.5
            if norm_a < 1e-12 or norm_b < 1e-12:
                return 0.0
            return sum(a * b for a, b in zip(embedding, doc_emb)) / (norm_a * norm_b)

        scored = [(sim(d["embedding"]), d) for d in self.docs.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]

    def get_all_docs(self) -> List[Dict[str, Any]]:
        return list(self.docs.values())


class EnhancedMockLLM:
    """LLM with richer simulation capabilities"""

    def __init__(self):
        self.hil_accept = False
        self.hil_action = HILAction.ACCEPT
        self.hil_modified_text = None
        self.hil_confidence = 1.0
        self.hil_notes = ""
        self.generation_count = 0

    def generate(self, prompt: str) -> str:
        self.generation_count += 1

        try:
            p = json.loads(prompt)
        except Exception:
            return json.dumps(
                {
                    "interpretation": prompt,
                    "delta_summary": prompt[:200],
                    "confidence": 0.5,
                }
            )

        role = p.get("role")

        if role == "provisional_interpreter":
            inp = p.get("input", "")
            ctx = p.get("context", "")

            # Simulate confidence based on input/context alignment
            confidence = 0.7 if "core" in ctx.lower() else 0.6

            interpretation = f"Interpreted: {inp[:80]}"
            if "contradict" in inp.lower():
                interpretation += " [POTENTIAL CONFLICT DETECTED]"
                confidence = 0.4

            return json.dumps(
                {
                    "interpretation": interpretation,
                    "delta_summary": interpretation[:150],
                    "confidence": confidence,
                }
            )

        elif role == "critic":
            interp = p.get("interpretation", "")
            ctx = p.get("context", "")

            # Simulate critique based on content
            critique = "Analysis: "
            if "[REFINED]" in interp:
                critique += (
                    "Previous refinement detected. Minor remaining alignment issues."
                )
            elif "CONFLICT" in interp:
                critique += (
                    "Contradiction with existing beliefs detected. Recommend alignment."
                )
            else:
                critique += "Semantic drift from context observed. Suggest tightening."

            return json.dumps({"critique": critique})

        elif role == "refine":
            prev = p.get("prev", "")
            critique = p.get("critique", "")

            # Simulate refinement
            refined = prev
            if "[REFINED]" not in refined:
                refined += " [REFINED]"
            else:
                refined = refined.replace("[REFINED]", "[REFINED×2]")

            if "alignment" in critique.lower():
                refined += " [ALIGNED]"

            return json.dumps({"refined": refined})

        elif role == "contradiction_detector":
            new = p.get("new", "")
            existing = p.get("existing", "")

            # Simple contradiction heuristic
            score = 0.0
            if "not" in new.lower() and any(
                word in existing.lower() for word in ["is", "are", "always"]
            ):
                score = 0.8
            elif set(new.lower().split()) & set(existing.lower().split()):
                score = 0.2  # Some overlap
            else:
                score = 0.5  # Uncertain

            return json.dumps({"contradiction_score": score})

        return json.dumps({"interpretation": str(prompt)})

    def simulate_hil(
        self,
        identity_summary: str,
        proposed_delta: str,
        evidence_snippets: List[Dict[str, Any]],
    ) -> bool:
        return self.hil_accept

    def simulate_hil_enhanced(
        self,
        identity_summary: str,
        proposed_delta: str,
        evidence_snippets: List[Dict[str, Any]],
        distortion: DistortionScore,
    ) -> HILResponse:
        """Enhanced HIL simulation with richer responses"""

        # If explicit action set, use it
        if self.hil_action == HILAction.MODIFY and self.hil_modified_text:
            return HILResponse(
                action=HILAction.MODIFY,
                modified_delta=self.hil_modified_text,
                confidence=self.hil_confidence,
                notes=self.hil_notes,
            )

        # Simulate decision based on distortion
        if distortion.composite > 0.8:
            action = HILAction.REJECT if not self.hil_accept else HILAction.ACCEPT
        elif distortion.uncertainty > 0.7:
            action = HILAction.DEFER
        else:
            action = HILAction.ACCEPT if self.hil_accept else HILAction.REJECT

        return HILResponse(
            action=action,
            confidence=self.hil_confidence,
            notes=self.hil_notes
            or f"Auto-decision: distortion={distortion.composite:.2f}",
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


class TestHarness:
    """Utility for running ECE tests"""

    @staticmethod
    def create_test_system(hil_accept: bool = False):
        """Create a complete test system"""
        from ece_engine_v2 import ECEEngine, ECEConfig, StructuredMemory

        llm = EnhancedMockLLM()
        llm.hil_accept = hil_accept

        embedder = EnhancedMockEmbedder()
        vectorstore = InMemoryVectorStore()
        memory = StructuredMemory()
        config = ECEConfig()

        engine = ECEEngine(llm, embedder, vectorstore, memory, config)

        return engine, llm, vectorstore, memory

    @staticmethod
    def seed_knowledge(engine, facts: List[str], stability_level: str = "working"):
        """Pre-populate the system with facts"""
        from ece_engine_v2 import BeliefStability

        stability_map = {
            "core": BeliefStability.CORE,
            "working": BeliefStability.WORKING,
            "ephemeral": BeliefStability.EPHEMERAL,
        }
        stability = stability_map.get(stability_level, BeliefStability.WORKING)

        for fact in facts:
            engine.commit_update(
                fact,
                f"seed:{fact[:30]}",
                metadata={"stability": stability.value, "source": "seed"},
            )

    @staticmethod
    def print_result(result: Dict[str, Any], verbose: bool = True):
        """Pretty print ECE result"""
        print(f"\n{'=' * 60}")
        print(f"STATUS: {result['status']}")
        print(f"{'=' * 60}")

        if verbose:
            print(f"\nInterpretation:")
            print(f"  {result['interpretation']}")

            if "distortion" in result:
                d = result["distortion"]
                print(f"\nDistortion Breakdown:")
                if hasattr(d, "semantic_distance"):
                    print(f"  Semantic:      {d.semantic_distance:.3f}")
                    print(f"  Contradiction: {d.contradiction_score:.3f}")
                    print(f"  Temporal:      {d.temporal_relevance:.3f}")
                    print(f"  Coherence:     {d.cluster_coherence:.3f}")
                    print(f"  Uncertainty:   {d.uncertainty:.3f}")
                    print(f"  COMPOSITE:     {d.composite:.3f}")
                else:
                    print(f"  Value: {d}")

            if "confidence" in result:
                print(f"\nConfidence: {result['confidence']:.3f}")

            if "refinement_iters" in result:
                print(f"Refinement iterations: {result['refinement_iters']}")

            if "hil_notes" in result:
                print(f"HIL Notes: {result['hil_notes']}")

        print(f"{'=' * 60}\n")


# Example usage
if __name__ == "__main__":
    # Create test system
    engine, llm, vectorstore, memory = TestHarness.create_test_system(hil_accept=True)

    # Seed with core beliefs
    TestHarness.seed_knowledge(
        engine,
        [
            "The system prioritizes accuracy over speed",
            "User privacy is fundamental and non-negotiable",
            "Transparency in decision-making is essential",
        ],
        stability_level="core",
    )

    # Test interpretation
    print("Testing normal update...")
    result = engine.haze_loop("New feature request: add logging")
    TestHarness.print_result(result)

    # Test contradictory input
    print("\nTesting contradictory input...")
    llm.set_hil_behavior(accept=False, notes="Contradicts privacy policy")
    result = engine.haze_loop("We should sell user data to advertisers")
    TestHarness.print_result(result)

    # Test modification
    print("\nTesting HIL modification...")
    llm.set_hil_behavior(
        action=HILAction.MODIFY,
        modified_text="Add privacy-preserving analytics with user consent",
        confidence=0.8,
        notes="Modified to align with privacy principles",
    )
    result = engine.haze_loop("Add analytics to track user behavior")
    TestHarness.print_result(result)

    print(f"\nTotal LLM calls: {llm.generation_count}")
    print(f"Documents in vectorstore: {len(vectorstore.docs)}")
