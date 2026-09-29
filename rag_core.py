import os
import re
import sys
from pathlib import Path

# Offline flags must be set before the ML libraries are imported
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
FROZEN = getattr(sys, "frozen", False)
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
MODEL_PATH = RESOURCE_DIR / "models" / "all-MiniLM-L6-v2"

if FROZEN:  # packaged .exe: keep the database somewhere writable
    DB_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "OfflineRAG" / "chroma_db"
else:
    DB_PATH = Path(__file__).parent / "chroma_db"

# ---------- Settings ----------
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_DISTANCE = 0.6      # cosine distance; lower = stricter. Tune with DEBUG.
DEBUG = False
SUPPORTED = {".pdf", ".docx", ".txt", ".csv"}
REFUSAL = ("I can only answer from your indexed documents, and I couldn't "
           "find relevant information there for that question.")

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(str(MODEL_PATH))
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = client.get_or_create_collection(
            name="documents",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------- Reading files ----------
def _extract(path: Path):
    """Return a list of (page_number, text). page_number is 0 when not applicable."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
    if ext == ".docx":
        doc = Document(str(path))
        return [(0, "\n".join(p.text for p in doc.paragraphs))]
    if ext == ".txt":
        return [(0, path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".csv":
        df = pd.read_csv(path).fillna("")
        rows = [" | ".join(f"{c}: {v}" for c, v in r.items()) for _, r in df.iterrows()]
        return [(0, "\n".join(rows))]
    raise ValueError(f"Unsupported file type: {ext}")


def _chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


# ---------- Public API ----------
def ingest_file(path):
    """Index one file. Re-adding the same filename replaces the old version."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Supported types: {', '.join(sorted(SUPPORTED))}")

    col = _get_collection()
    col.delete(where={"source": path.name})

    docs, metas, ids = [], [], []
    for page, text in _extract(path):
        for i, chunk in enumerate(_chunk(text)):
            docs.append(chunk)
            metas.append({"source": path.name, "page": page})
            ids.append(f"{path.name}::{page}::{i}")
    if not docs:
        return 0

    embeddings = _get_embedder().encode(
        docs, normalize_embeddings=True, batch_size=32).tolist()
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(docs)


def list_documents():
    data = _get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def remove_document(name):
    """Remove a file from the index. Returns how many chunks were deleted."""
    col = _get_collection()
    n = len(col.get(where={"source": name}, include=[])["ids"])
    if n:
        col.delete(where={"source": name})
    return n


def ask(question):
    """Return (answer, sources). Refuses when nothing relevant is indexed."""
    col = _get_collection()
    if col.count() == 0:
        return "No documents are indexed yet. Add some files first.", []

    q_emb = _get_embedder().encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb,
                    n_results=min(TOP_K, col.count()),
                    include=["documents", "metadatas", "distances"])
    hits = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
    if DEBUG:
        print("distances:", [round(d, 3) for _, _, d in hits])

    hits = [h for h in hits if h[2] <= MAX_DISTANCE]
    if not hits:
        return REFUSAL, []

    def label(m):
        return f"{m['source']}, p.{m['page']}" if m["page"] else m["source"]

    context = "\n\n".join(f"[{label(m)}]\n{d}" for d, m, _ in hits)
    system = ("You answer questions using ONLY the context provided. "
              "If the answer is not in the context, reply with exactly: NOT_IN_DOCUMENTS. "
              "Never use outside knowledge. Be concise.")
    user = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": 0.1, "num_predict": 220, "num_ctx": 2048},
            keep_alive="30m",
        )
    except Exception as e:
        return (f"Could not reach Ollama. Make sure it is running "
                f"(llama icon in the system tray). Details: {e}"), []

    answer = resp["message"]["content"].strip()
    if answer.startswith("NOT_IN_DOCUMENTS"):
        return REFUSAL, []

    pages = {}
    for _, m, _ in hits:
        pages.setdefault(m["source"], set())
        if m["page"]:
            pages[m["source"]].add(m["page"])
    sources = []
    for src, pgs in pages.items():
        if pgs:
            sources.append(f"{src}, p.{', '.join(str(p) for p in sorted(pgs))}")
        else:
            sources.append(src)
    return answer, sources


def warmup():
    """Load the LLM into memory ahead of time so the first real question is fast."""
    try:
        ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
            keep_alive="30m",
        )
    except Exception:
        pass