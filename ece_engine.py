# eidetic_continuum/ece_engine.py
from typing import List, Dict, Any, Optional
import time
import json
import math

# Interfaces expected from injected components:
# llm: object with method generate(prompt:str)->str
# embedder: object with method embed(text:str)->List[float]
# vectorstore: object with methods
#    add_document(id:str, text:str, embedding:List[float], metadata:dict)
#    query_by_embedding(embedding:List[float], k:int)->List[dict] each dict {id,text,embedding,metadata}
# memory: object with methods
#    get_summary()->str
#    update_with(delta_text:str, source_meta:dict)->None


def cosine_sim(a, b):
    na = math.sqrt(sum(x * x for x in a)) + 1e-12
    nb = math.sqrt(sum(x * x for x in b)) + 1e-12
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class ECEConfig:
    def __init__(
        self,
        k: int = 5,
        tau_refine: float = 0.35,
        tau_accept: float = 0.20,
        tau_hil: float = 0.60,
        max_refine_iters: int = 3,
    ):
        self.k = k
        self.tau_refine = tau_refine
        self.tau_accept = tau_accept
        self.tau_hil = tau_hil
        self.max_refine_iters = max_refine_iters


class ECEEngine:
    def __init__(self, llm, embedder, vectorstore, memory, config: ECEConfig = None):
        self.llm = llm
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.memory = memory
        self.config = config or ECEConfig()

    # small JSON-based prompt/response helpers for deterministic mocks
    def _provisional_prompt(self, input_text: str, context_summary: str) -> str:
        return json.dumps(
            {
                "role": "provisional_interpreter",
                "input": input_text,
                "context": context_summary,
            }
        )

    def _critic_prompt(self, interpretation: str, context_summary: str) -> str:
        return json.dumps(
            {
                "role": "critic",
                "interpretation": interpretation,
                "context": context_summary,
            }
        )

    def _refine_prompt(self, prev: str, critique: str, context: str) -> str:
        return json.dumps(
            {"role": "refine", "prev": prev, "critique": critique, "context": context}
        )

    def _parse_provisional(self, raw: str) -> Dict[str, str]:
        # Expecting JSON with fields interpretation, delta_summary
        try:
            j = json.loads(raw)
            # for real LLM replace parsing with robust method
            interpretation = j.get("interpretation") or j.get("result") or raw
            delta_summary = (
                j.get("delta_summary") or j.get("delta") or interpretation[:200]
            )
            return {"interpretation": interpretation, "delta_summary": delta_summary}
        except Exception:
            return {"interpretation": raw, "delta_summary": raw[:200]}

    def measure_distortion(
        self, interpretation_text: str, context_docs: List[Dict[str, Any]]
    ) -> float:
        e_i = self.embedder.embed(interpretation_text)
        sims = []
        for d in context_docs:
            e_d = d.get("embedding")
            if e_d is None:
                e_d = self.embedder.embed(d.get("text", ""))
            sims.append(cosine_sim(e_i, e_d))
        max_sim = max(sims) if sims else 0.0
        semantic_distortion = 1.0 - max_sim
        # future: add NLI contradiction detector and entropy-based novelty
        return semantic_distortion

    def commit_update(
        self, delta_text: str, input_text: str, metadata: Optional[dict] = None
    ):
        meta = metadata or {}
        meta.update({"time": time.time()})
        doc_id = f"patch_{int(time.time() * 1000)}"
        emb = self.embedder.embed(delta_text)
        self.vectorstore.add_document(doc_id, delta_text, emb, meta)
        # update memory summary (simple append + memory internal summarizer)
        self.memory.update_with(delta_text, {"origin_input": input_text, **meta})

    def human_in_loop_decision(
        self,
        identity_summary: str,
        proposed_delta: str,
        evidence_snippets: List[Dict[str, Any]],
    ) -> bool:
        # This is a placeholder: in production you'd show a UI to an operator.
        # For the tests/mocks we assume the injected 'llm' or a test harness controls this.
        # We'll default to False (reject) to be safe.
        # But if llm has a 'simulate_hil' hook we can call it.
        if hasattr(self.llm, "simulate_hil"):
            return self.llm.simulate_hil(
                identity_summary, proposed_delta, evidence_snippets
            )
        return False

    def haze_loop(self, input_text: str) -> Dict[str, Any]:
        # 1) retrieve context
        context_summary = self.memory.get_summary()
        # query by embedding of input_text
        emb_input = self.embedder.embed(input_text)
        docs = self.vectorstore.query_by_embedding(emb_input, k=self.config.k)
        # 2) provisional interpret
        prompt = self._provisional_prompt(input_text, context_summary)
        raw_out = self.llm.generate(prompt)
        prov = self._parse_provisional(raw_out)
        interpretation = prov["interpretation"]
        # 3) measure distortion
        distortion = self.measure_distortion(interpretation, docs)
        if distortion >= self.config.tau_hil:
            accepted = self.human_in_loop_decision(
                context_summary, interpretation, docs[: self.config.k]
            )
            if not accepted:
                return {
                    "status": "rejected_by_human",
                    "interpretation": interpretation,
                    "distortion": distortion,
                }
            self.commit_update(
                interpretation, input_text, metadata={"distortion": distortion}
            )
            return {
                "status": "committed_by_human",
                "interpretation": interpretation,
                "distortion": distortion,
            }
        if distortion >= self.config.tau_refine:
            # self-refine
            prev = interpretation
            iters = 0
            while iters < self.config.max_refine_iters:
                crit_prompt = self._critic_prompt(prev, context_summary)
                critique = self.llm.generate(crit_prompt)
                refine_prompt = self._refine_prompt(prev, critique, context_summary)
                refined_raw = self.llm.generate(refine_prompt)
                # accept refined as new interpretation
                prev = (
                    json.loads(refined_raw).get("refined")
                    if refined_raw.strip().startswith("{")
                    else refined_raw
                )
                distortion_new = self.measure_distortion(prev, docs)
                iters += 1
                if distortion_new < self.config.tau_refine:
                    break
            interpretation = prev
        final_distortion = self.measure_distortion(interpretation, docs)
        if final_distortion <= self.config.tau_accept:
            self.commit_update(
                interpretation, input_text, metadata={"distortion": final_distortion}
            )
            return {
                "status": "auto_committed",
                "interpretation": interpretation,
                "distortion": final_distortion,
            }
        else:
            accepted = self.human_in_loop_decision(
                context_summary, interpretation, docs[: self.config.k]
            )
            if accepted:
                self.commit_update(
                    interpretation,
                    input_text,
                    metadata={"distortion": final_distortion},
                )
                return {
                    "status": "committed_by_human_post_refine",
                    "interpretation": interpretation,
                    "distortion": final_distortion,
                }
            return {
                "status": "rejected",
                "interpretation": interpretation,
                "distortion": final_distortion,
            }
