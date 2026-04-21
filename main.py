import os
import PyPDF2
import chromadb
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google import genai
import ollama
from pydantic import BaseModel

# --- KONFIGURASI ---
# Silakan isi API Key Gemini kamu di sini
GOOGLE_API_KEY = "AIzaSyDfRJi2IF9n-qwAjTjaTUNoPMpUy1YKpok"
OLLAMA_MODEL = "glm-5.1:cloud"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "expert_system_docs"

app = FastAPI()
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# Setup Static Files (untuk melayani HTML/CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- LOGIKA RAG ---
class GeminiEmbeddingFunction:
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            response = gemini_client.models.embed_content(model="text-embedding-004", contents=text)
            embeddings.append(response.embeddings[0].values)
        return embeddings

class RAGChatbot:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embed_fn = GeminiEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn
        )

    def process_pdf(self, file):
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        chunks = []
        chunk_size, overlap = 500, 50
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def ask(self, query):
        results = self.collection.query(query_texts=[query], n_results=3)
        context = "\n".join(results["documents"][0])
        
        system_prompt = (
            f"Kamu adalah asisten ahli. Gunakan HANYA informasi dari KONTEKS di bawah.\n"
            f"Jika tidak ada di konteks, katakan 'Maaf, tidak ditemukan di dokumen'.\n\n"
            f"KONTEKS:\n{context}\n\nPERTANYAAN: {query}\n\nJAWABAN:"
        )
        response = ollama.generate(model=OLLAMA_MODEL, prompt=system_prompt)
        return response['response']

# Inisialisasi Bot Global
bot = RAGChatbot()

# --- ENDPOINTS API ---

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    num_chunks = bot.process_pdf(file.file)
    return {"status": "success", "message": f"Berhasil memproses {num_chunks} chunk teks."}

@app.post("/chat")
async def chat(data: dict): # data = {"query": "pertanyaan user"}
    query = data.get("query")
    answer = bot.ask(query)
    return {"answer": answer}

@app.post("/reset")
async def reset_db():
    import shutil
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    # Re-initialize bot to refresh the collection
    global bot
    bot = RAGChatbot()
    return {"status": "success", "message": "Database berhasil dikosongkan."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
