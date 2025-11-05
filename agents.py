# eidetic_continuum/agents.py
# Simple mock implementations for LLM, Embedder, InMemoryVectorStore, and Memory.
# These are intentionally small and deterministic for unit tests.

import json
from typing import List, Dict, Any


class MockEmbedder:
    def __init__(self):
        # naive char-sum embedding to keep deterministic behavior
        pass

    def embed(self, text: str) -> List[float]:
        s = sum(ord(c) for c in text) % 1000
        # small vector of length 8 with repeated s/1000 pattern
        return [(s / 1000.0) for _ in range(8)]


class InMemoryVectorStore:
    def __init__(self):
        self.docs = {}  # id -> dict{text,embedding,metadata}

    def add_document(
        self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any]
    ):
        self.docs[id] = {
            "id": id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }

    def query_by_embedding(self, embedding: List[float], k: int = 5):
        # simple similarity via dot of repeated vectors
        def sim(doc_emb):
            return sum(a * b for a, b in zip(doc_emb, embedding))

        scored = [(sim(d["embedding"]), d) for d in self.docs.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:k]]


class SimpleMemory:
    def __init__(self):
        self.summary = ""

    def get_summary(self) -> str:
        return self.summary

    def update_with(self, delta_text: str, source_meta: Dict[str, Any]):
        # naive append
        if not self.summary:
            self.summary = delta_text
        else:
            self.summary = self.summary + "\n" + delta_text


class MockLLM:
    def __init__(self):
        # ability to control human-in-loop
        self.hil_accept = False

    def generate(self, prompt: str) -> str:
        try:
            p = json.loads(prompt)
        except Exception:
            # fallback: echo
            return json.dumps({"interpretation": prompt, "delta_summary": prompt[:200]})
        role = p.get("role")
        if role == "provisional_interpreter":
            # create a deterministic interpretation
            inp = p.get("input", "")
            ctx = p.get("context", "")
            interpretation = f"INTERPRETATION: summarized [{inp[:60]}]"
            return json.dumps(
                {"interpretation": interpretation, "delta_summary": interpretation}
            )
        if role == "critic":
            return json.dumps({"critique": "minor mismatch with context"})
        if role == "refine":
            # produce a refined JSON with 'refined' key
            prev = p.get("prev", "")
            refined = prev + " [REFINED]"
            return json.dumps({"refined": refined})
        return json.dumps({"interpretation": str(prompt)})

    def simulate_hil(self, identity_summary, proposed_delta, evidence_snippets):
        # return the preset choice
        return self.hil_accept
