\# Offline RAG Assistant



A fully offline document Q\&A desktop app for Windows. Ask questions about your own PDF, DOCX, TXT, or CSV files — answered locally with page-level citations, no internet required at query time.



\## Stack

\- Ollama (llama3.2:3b) — local LLM

\- ChromaDB — vector store

\- Sentence-Transformers (all-MiniLM-L6-v2) — embeddings

\- CustomTkinter — GUI

\- PyInstaller — packaging



\## Setup

1\. Install \[Ollama](https://ollama.com) and run `ollama pull llama3.2:3b`

2\. `python -m venv venv \&\& venv\\Scripts\\activate`

3\. `pip install -r requirements.txt`

4\. `python app.py`

