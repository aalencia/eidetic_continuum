Eidetic Continuum Engine - quick run

1) Create project layout:
   mkdir ece_project
   cd ece_project
   mkdir eidetic_continuum tests

2) Save files:
   - eidetic_continuum/ece_engine.py  (paste content)
   - eidetic_continuum/agents.py      (paste content)
   - tests/test_ece_engine.py         (paste content)

3) Create virtualenv and install pytest:
   python -m venv .venv
   source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
   pip install pytest

4) Run tests:
   pytest -q

The tests use only builtin libs and the modules above. No external APIs or keys needed.

SWITCHING TO REAL LLM / VECTORSTORE:
- Replace `MockLLM` with your LangChain/OpenAI/local LLM wrapper that exposes `.generate(prompt)->str`.
- Replace `MockEmbedder` with the embeddings provider (OpenAIEmbeddings or similar).
- Replace `InMemoryVectorStore` with Chroma/FAISS/Pinecone wrapper implementing `add_document` and `query_by_embedding`.
- Keep `SimpleMemory` or replace with LangChain ConversationSummaryMemory, calling its summary API.

TUNING:
- Adjust ECEConfig thresholds: tau_refine, tau_accept, tau_hil, k
- Implement a real `measure_distortion` adding contradiction detection (NLI) / novelty.
