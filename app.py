import os, json, httpx
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_utils import load_kb, retrieve, load_embedder, load_chroma
from fastapi.middleware.cors import CORSMiddleware
# Ollama local server config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME  = os.getenv("LOCAL_MODEL", "llama3.1")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://c137a2849114.ngrok-free.app",  # add once you deploy
        "http://localhost:3000",                     # optional dev
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory="static"), name="static")

kb_df = None
index = load_chroma("db/chroma")
id_map = None
embedder = None

@app.on_event("startup")
async def startup():
    global kb_df, index, id_map, embedder
    kb_df = load_kb("db/medical_kb.csv")
    # If you’re using FAISS:
    #index, id_map = None
    # If you switched to Chroma earlier, instead do:
    from rag_utils import load_chroma
    index = load_chroma("db/chroma")
    embedder = load_embedder()

@app.get("/")
def home():
    return FileResponse("static/index.html")

class ChatPayload(BaseModel):
    messages: List[Dict[str, Any]]

MEDICAL_GUARD = (
    "You are a medical information assistant for general education only. "
    "Answer ONLY medical/health questions using the provided sources; add short citations like [source (section)]. "
    "Do NOT provide diagnosis or treatment advice. Encourage seeing a clinician for personal cases. "
    "If asked non-medical questions, refuse and say you only answer medical topics."
)

async def ollama_stream(messages: List[Dict[str, str]]):
    async with httpx.AsyncClient(timeout=60.0) as client:
        req = {"model": MODEL_NAME, "stream": True, "messages": messages}
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=req) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise HTTPException(status_code=500, detail=text.decode("utf-8"))
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield f"data:{token}\n\n"
            yield "data:[DONE]\n\n"

@app.post("/chat")
async def chat(payload: ChatPayload, request: Request):
    user_text = ""
    for m in reversed(payload.messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    # basic scope guard
    if any(x in user_text.lower() for x in ["weather","stock","movie","sports","travel","python","finance"]):
        def _refusal():
            yield "data: I can only answer medical/health questions.\n\n"
            yield "data:[DONE]\n\n"
        return StreamingResponse(_refusal(), media_type="text/event-stream")

    # RAG retrieve
    ctx_docs = retrieve(user_text, k=4, index=index, id_map=id_map, embedder=embedder, kb_df=kb_df)
    ctx_text = "\n\n".join([f"[{d['source']} ({d['section']})]\n{d['text']}" for d in ctx_docs])

    convo = [
        {"role":"system","content": MEDICAL_GUARD},
        {"role":"system","content": f"Context:\n{ctx_text}"}
    ] + payload.messages

    return StreamingResponse(ollama_stream(convo), media_type="text/event-stream")
