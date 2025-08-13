import os
import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def load_kb(csv_path="db/medical_kb.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

def build_chroma(kb_df: pd.DataFrame, persist_dir="db/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    # recreate collection each time
    try:
        client.delete_collection("medical")
    except Exception:
        pass
    collection = client.create_collection(name="medical")
    model = load_embedder()
    texts = kb_df["text"].tolist()
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    ids = kb_df["id"].astype(str).tolist()
    metas = kb_df[["source","section"]].to_dict(orient="records")
    collection.add(ids=ids, embeddings=embs.tolist(), documents=texts, metadatas=metas)
    return persist_dir

def load_chroma(persist_dir="db/chroma"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection("medical")

def retrieve(query: str, k: int = 4, index=None, id_map=None, embedder=None, kb_df=None):
    # Here, 'index' will be the Chroma collection
    if embedder is None:
        embedder = load_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    res = index.query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],  # ids are returned by default in many versions; if not, we can omit
    )

    out = []
    ids = res.get("ids", [[]])[0] if "ids" in res else [None] * len(res["documents"][0])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    for i in range(len(docs)):
        out.append({
            "id": ids[i] if ids else None,
            "text": docs[i],
            "source": metas[i].get("source", "unknown"),
            "section": metas[i].get("section", "unknown"),
        })
    return out