import os
import shutil
import PyPDF2
import chromadb
import ollama
import json
import urllib.request
import urllib.error
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
COLLECTION_NAME_BASE = "expert_system_docs"

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Konfigurasi Layanan Lokal
SECRET_API_KEY = os.getenv("SECRET_API_KEY", "g4-rahasia")
OLLAMA_MODEL = ""
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

# --- EMBEDDING LOGIC (HYBRID) ---
# --- EMBEDDING LOGIC (HYBRID) ---
# --- EMBEDDING LOGIC (HYBRID) ---
class HybridEmbeddingFunction:
    def __init__(self, mode="local", api_key=None):
        self.mode = mode
        self.api_key = api_key

    def name(self):
        """Method wajib yang dicari oleh ChromaDB untuk validasi koleksi"""
        return f"hybrid-embedding-{self.mode}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # ChromaDB memanggil fungsi ini saat .add()
        if self.mode == "cloud" and self.api_key:
            return self.embed_cloud(input)
        return self.embed_local(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Kompatibilitas untuk ChromaDB/Langchain saat menyimpan dokumen"""
        return self.__call__(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Kompatibilitas wajib untuk ChromaDB saat melakukan pencarian (.query)"""
        return self.__call__(input)

    def embed_local(self, input: list[str]) -> list[list[float]]:
        """Menggunakan Ollama untuk Embedding (Dimensi: 896 untuk Qwen2.5:0.5b)"""
        embeddings = []
        for text in input:
            try:
                response = local_client.embeddings(model=OLLAMA_MODEL, prompt=text)
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"❌ Error Local Embedding: {e}")
                embeddings.append([0.0] * 896)
        return embeddings

    def embed_cloud(self, input: list[str]) -> list[list[float]]:
        """Menggunakan Gemini API untuk Embedding (Dimensi: 768)"""
        embeddings = []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        for text in input:
            try:
                payload = {
                    "model": "models/text-embedding-004", 
                    "content": {"parts": [{"text": text}]}
                }
                req = urllib.request.Request(
                    url, 
                    data=json.dumps(payload).encode('utf-8'), 
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req) as res:
                    data = json.loads(res.read().decode('utf-8'))
                    embeddings.append(data['embedding']['values'])
            except Exception as e:
                print(f"❌ Error Cloud Embedding: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

# --- RAG ENGINE ---
class RAGChatbot:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    def get_collection(self, auth_ctx):
        """Membuat/Mengambil koleksi berdasarkan mode agar dimensi vektor tidak bentrok"""
        mode = auth_ctx["mode"]
        api_key = auth_ctx.get("key")
        
        # Nama koleksi dibedakan: expert_system_docs_local atau expert_system_docs_cloud
        col_name = f"{COLLECTION_NAME_BASE}_{mode}"
        embed_fn = HybridEmbeddingFunction(mode=mode, api_key=api_key)
        
        return self.chroma_client.get_or_create_collection(
            name=col_name,
            embedding_function=embed_fn
        )

    def process_pdf(self, file_path, filename, auth_ctx):
        session_id = auth_ctx["session_id"]
        collection = self.get_collection(auth_ctx)

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
            metadatas = [{"session_id": session_id, "filename": filename} for _ in chunks]
            collection.add(documents=chunks, ids=ids, metadatas=metadatas)
            return len(chunks)
        return 0

    def delete_document(self, filename, auth_ctx):
        collection = self.get_collection(auth_ctx)
        try:
            collection.delete(where={"$and": [{"session_id": auth_ctx["session_id"]}, {"filename": filename}]})
            return True
        except:
            return False

    def ask(self, query, history, language, auth_ctx):
        session_id = auth_ctx["session_id"]
        mode = auth_ctx["mode"]
        collection = self.get_collection(auth_ctx)
        
        # 1. RETRIEVAL DENGAN ISOLASI SESI & MODE
        results = collection.query(
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
            f"Anda adalah G4 Expert System, seorang analis data yang sangat teliti.\n"
            f"ATURAN MUTLAK:\n"
            f"1. Anda WAJIB menjawab HANYA berdasarkan informasi pada KONTEKS DOKUMEN di bawah.\n"
            f"2. Jika jawaban tidak ditemukan dalam konteks, Anda harus menjawab: 'Maaf, informasi tersebut tidak ditemukan dalam basis pengetahuan dokumen Anda.'\n"
            f"3. Jangan pernah menebak atau memberikan informasi fiktif.\n\n"
            f"KONTEKS DOKUMEN:\n{context if context else 'Belum ada dokumen.'}\n\n"
            f"RIWAYAT:\n{history_text}\n"
            f"PERTANYAAN: {query}\n\n"
            f"Berikan jawaban analitis dalam {target_lang}."
        )

        # 3. DYNAMIC ROUTING (LOCAL vs CLOUD) UNTUK GENERASI TEKS
        try:
            if mode == "local":
                response = local_client.generate(model=OLLAMA_MODEL, prompt=system_prompt)
                return response['response']
            else:
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

        # Kirim seluruh auth_ctx, bukan hanya session_id
        num_chunks = bot.process_pdf(temp_path, file.filename, auth_ctx)
        
        return {
            "status": "success", 
            "num_chunks": num_chunks, 
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/delete_file")
async def delete_file(data: DeleteRequest, auth_ctx: dict = Depends(get_auth_context)):
    print(f"DEBUG: Mencoba menghapus {data.filename} untuk sesi {auth_ctx['session_id']} (Mode: {auth_ctx['mode']})")
    success = bot.delete_document(data.filename, auth_ctx)
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