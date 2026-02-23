from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    id: str
    text: str

class EmbeddingResponse(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the text embedding API. Use the /embed endpoint to get embeddings for your text."}

@app.post("/embed")
def embed_text(req: TextRequest):
    embedding = model.encode(req.text).tolist()
    return {"id": req.id, "embedding": embedding}


@app.post('/query-to-embedding')
def query_to_embedding(req: EmbeddingResponse):
    embedding = model.encode(req.text).tolist()
    return { "embedding": embedding}