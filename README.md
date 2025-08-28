# Local Medical RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that runs **completely offline** on your laptop using [Ollama](https://ollama.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Chroma](https://www.trychroma.com/).  
It answers **medical/health questions only**, cites sources, and includes disclaimers — powered by a custom medical knowledge base.


## Features
-  **Local LLM with Ollama** — no data leaves our machine
- **Semantic search** with vector embeddings (finds meaning, not just keywords)
- **Retrieval-Augmented Generation (RAG)** — every answer is grounded in our unique dataset
- **Medical guardrails** — refuses non-medical queries, always cites sources, includes disclaimers
- **Web interface** — simple chat page with live streaming responses
