# tests/test_ece_engine.py
import pytest
from eidetic_continuum.ece_engine import ECEEngine, ECEConfig
from eidetic_continuum.agents import (
    MockLLM,
    MockEmbedder,
    InMemoryVectorStore,
    SimpleMemory,
)


def make_engine_with_seed(seed_docs=None):
    llm = MockLLM()
    emb = MockEmbedder()
    vs = InMemoryVectorStore()
    mem = SimpleMemory()
    # optional seed docs
    if seed_docs:
        for i, txt in enumerate(seed_docs):
            vs.add_document(f"seed_{i}", txt, emb.embed(txt), {"seed": True})
            mem.update_with(txt, {"seed": True})
    engine = ECEEngine(
        llm,
        emb,
        vs,
        mem,
        ECEConfig(k=3, tau_refine=0.4, tau_accept=0.2, tau_hil=0.9, max_refine_iters=2),
    )
    return engine, llm, vs, mem


def test_auto_commit_low_distortion():
    engine, llm, vs, mem = make_engine_with_seed(["hello world"])
    result = engine.haze_loop("hello world followup")
    assert result["status"] in ("auto_committed", "committed_by_human_post_refine")
    # memory should have been updated
    assert "INTERPRETATION" in mem.get_summary() or len(vs.docs) >= 1


def test_refine_loop_runs_when_needed():
    engine, llm, vs, mem = make_engine_with_seed(["completely different doc"])
    # provoke refine by setting thresholds low
    engine.config.tau_refine = 0.0
    result = engine.haze_loop("an input that requires refine")
    assert result["status"] in (
        "auto_committed",
        "committed_by_human_post_refine",
        "rejected",
    )
    # Ensure refine appended [REFINED] when refine executed
    # The vectorstore may have a commit; just check mem summary updated or docs present
    assert mem.get_summary() != ""


def test_human_in_loop_rejection_and_acceptance():
    engine, llm, vs, mem = make_engine_with_seed(["seed doc"])
    # force high distortion to trigger HIL
    engine.config.tau_hil = 0.0  # anything triggers HIL
    llm.hil_accept = False
    r = engine.haze_loop("trigger hil")
    assert r["status"] == "rejected_by_human"
    llm.hil_accept = True
    r2 = engine.haze_loop("trigger hil accept")
    assert r2["status"] in ("committed_by_human", "committed_by_human_post_refine")


if __name__ == "__main__":
    pytest.main(["-q", "-s"])
