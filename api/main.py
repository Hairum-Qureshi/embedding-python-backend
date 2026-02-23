import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Get a free token from: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"

class TextRequest(BaseModel):
    id: str
    text: str

class EmbeddingResponse(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Embedding API for the CIS advisor chat bot!"}

@app.post("/embed")
async def embed_text(req: TextRequest):
    return {"id": req.id, "embedding": await get_hf_embedding(req.text)}

@app.post("/query-to-embedding")
async def query_to_embedding(req: EmbeddingResponse):
    return {"embedding": await get_hf_embedding(req.text)}

async def get_hf_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, headers=headers, json={"inputs": text})
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Hugging Face API Error")
        
    return response.json()