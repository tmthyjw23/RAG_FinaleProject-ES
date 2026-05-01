import os
import shutil
import PyPDF2
import chromadb
import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
COLLECTION_NAME = "expert_system_docs"

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Konfigurasi G4 Local Service
SECRET_API_KEY = os.getenv("SECRET_API_KEY", "g4-rahasia")
OLLAMA_MODEL = "qwen2.5:0.5b"
local_client = ollama.Client(host='http://localhost:11434')

# --- SECURITY & ROUTING DEPENDENCY ---
def get_auth_context(
    authorization: str = Header(None),
    x_service_mode: str = Header("local"),
    x_session_id: str = Header(...)
):
    """Mengecek otorisasi berdasarkan mode layanan dan mengambil ID Sesi user"""
    token = authorization.replace("Bearer ", "") if authorization else ""
    
    if x_service_mode == "local":
        if token != SECRET_API_KEY:
            raise HTTPException(status_code=401, detail="Password G4 Local tidak valid.")
        return {"mode": "local", "key": None, "session_id": x_session_id}
    
    elif x_service_mode == "cloud":
        if not token:
            raise HTTPException(status_code=401, detail="API Key Cloud GenAI diperlukan.")
        return {"mode": "cloud", "key": token, "session_id": x_session_id}
    
    raise HTTPException(status_code=400, detail="Mode layanan tidak dikenali.")

# --- EMBEDDING LOGIC ---
class HybridEmbeddingFunction:
    def name(self):
        return "hybrid-embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # ChromaDB memanggil fungsi ini secara langsung
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Logika utama untuk mengubah teks menjadi vektor"""
        embeddings = []
        for text in input:
            try:
                # Menggunakan Ollama untuk Embedding
                response = local_client.embeddings(model=OLLAMA_MODEL, prompt=text)
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"❌ Error Embedding: {e}")
                # Dimensi 896 sesuai untuk Qwen2.5:0.5b
                embeddings.append([0.0] * 896)
        return embeddings

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Metode wajib yang dicari ChromaDB saat menjalankan .query()"""
        # Arahkan logika embed query kembali ke embed_documents
        return self.embed_documents(input)

# --- RAG ENGINE ---
class RAGChatbot:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embed_fn = HybridEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn
        )

    def process_pdf(self, file_path, filename, session_id):
        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 800:
                current_chunk += p + "\n"
            else:
                if current_chunk.strip(): chunks.append(current_chunk.strip())
                current_chunk = p + "\n"
        if current_chunk.strip(): chunks.append(current_chunk.strip())
        chunks = [c for c in chunks if len(c) > 15]

        if chunks:
            ids = [f"doc_{session_id}_{os.urandom(4).hex()}_{i}" for i in range(len(chunks))]
            # Menambahkan METADATA untuk isolasi user dan nama file
            metadatas = [{"session_id": session_id, "filename": filename} for _ in chunks]
            self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            return len(chunks)
        return 0

    def delete_document(self, filename, session_id):
        # Menghapus secara presisi berdasarkan kepemilikan dan nama file
        try:
            self.collection.delete(where={"$and": [{"session_id": session_id}, {"filename": filename}]})
            return True
        except:
            return False

    def ask(self, query, history, language, auth_ctx):
        session_id = auth_ctx["session_id"]
        mode = auth_ctx["mode"]
        
        # 1. RETRIEVAL DENGAN ISOLASI SESI
        results = self.collection.query(
            query_texts=[query], 
            n_results=5,
            where={"session_id": session_id} 
        )

        context = ""
        if results["documents"] and results["documents"][0]:
            context = "\n---\n".join(results["documents"][0])

        # 2. MEMBANGUN INGATAN (MEMORY)
        history_text = ""
        for msg in history[-4:]:
            role = "Sistem Pakar" if msg["role"] == "bot" else "Pengguna"
            history_text += f"{role}: {msg['text']}\n"

        target_lang = "bahasa Indonesia" if language == "Indonesia" else ("English" if language == "English" else "Mandarin (Chinese)")

        system_prompt = (
            f"Anda adalah Sistem Pakar yang disiplin. Gunakan KONTEKS dokumen berikut untuk menjawab jika relevan. "
            f"Jika tidak ada di konteks, Anda boleh menggunakan pengetahuan umum Anda tapi beritahu bahwa itu bukan dari dokumen.\n\n"
            f"KONTEKS DOKUMEN:\n{context if context else 'Belum ada dokumen yang diunggah pengguna.'}\n\n"
            f"RIWAYAT OBROLAN TERAKHIR:\n{history_text}\n"
            f"PERTANYAAN BARU: {query}\n\n"
            f"Jawablah dengan {target_lang} yang baik."
        )

        # 3. DYNAMIC ROUTING (LOCAL vs CLOUD)
        try:
            if mode == "local":
                response = local_client.generate(model=OLLAMA_MODEL, prompt=system_prompt)
                return response['response']
            else:
                import urllib.request
                import urllib.error
                import json
                
                api_key = auth_ctx["key"]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                
                payload = {
                    "contents": [{"parts": [{"text": system_prompt}]}],
                    "generationConfig": {"temperature": 0.3}
                }
                
                req = urllib.request.Request(
                    url, 
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                
                try:
                    with urllib.request.urlopen(req) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        if "candidates" in result and len(result["candidates"]) > 0:
                            return result["candidates"][0]["content"]["parts"][0]["text"]
                        return "⚠️ Respon diblokir oleh filter keamanan AI atau struktur tidak valid."
                except urllib.error.HTTPError as e:
                    error_body = json.loads(e.read().decode('utf-8'))
                    error_msg = error_body.get('error', {}).get('message', str(e))
                    return f"❌ Error Gemini API: {error_msg}"
                    
        except Exception as e:
            return f"❌ Error LLM ({mode}): {str(e)}"

bot = RAGChatbot()

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    language: str = "English"
    history: list = []
    
class DeleteRequest(BaseModel):
    filename: str

# --- API ENDPOINTS ---
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), auth_ctx: dict = Depends(get_auth_context)):
    temp_path = os.path.join(BASE_DIR, f"temp_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        num_chunks = bot.process_pdf(temp_path, file.filename, auth_ctx["session_id"])
        
        return {
            "status": "success", 
            "num_chunks": num_chunks, 
            "filename": file.filename
        }
    except Exception as e:
        # Melempar HTTPException agar frontend mengenali ini sebagai error (bukan 200 OK)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Mencegah penumpukan file sampah jika terjadi error
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/delete_file")
async def delete_file(data: DeleteRequest, auth_ctx: dict = Depends(get_auth_context)):
    print(f"DEBUG: Mencoba menghapus {data.filename} untuk sesi {auth_ctx['session_id']}")
    success = bot.delete_document(data.filename, auth_ctx["session_id"])
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=500, detail="Gagal menghapus file di database.")

@app.post("/chat")
async def chat(data: ChatRequest, auth_ctx: dict = Depends(get_auth_context)):
    if not data.query:
        return {"answer": "Silakan masukkan pertanyaan."}
    answer = bot.ask(data.query, data.history, data.language, auth_ctx)
    return {"answer": answer}

@app.get("/validate_local")
async def validate_local(auth_ctx: dict = Depends(get_auth_context)):
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)