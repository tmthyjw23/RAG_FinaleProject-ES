import os
import shutil
import PyPDF2
import chromadb
import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()
app = FastAPI()

# Enable CORS untuk frontend lokal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi Client Ollama (Mengarah ke WSL localhost)
client = ollama.Client(host='http://localhost:11434')

# Konfigurasi Model & Path
# Pastikan sudah: ollama pull qwen2.5-coder:3b
OLLAMA_MODEL = "qwen2.5:0.5b"

# Dapatkan lokasi absolut dari file main.py saat ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Gabungkan dengan nama folder chroma_db
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db") 

COLLECTION_NAME = "expert_system_docs"

# Setup Static Files untuk Frontend
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- LOGIKA EMBEDDING (KONSISTEN) ---
class OllamaEmbeddingFunction:
    """Menggunakan Ollama untuk menghasilkan vektor dengan metode yang diminta ChromaDB"""
    
    def name(self):
        return "ollama-embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Metode ini biasanya digunakan saat menambahkan dokumen (add)
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Logika untuk memproses banyak dokumen sekaligus"""
        embeddings = []
        for text in input:
            try:
                response = client.embeddings(model=OLLAMA_MODEL, prompt=text)
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"❌ Error Embedding Documents: {e}")
                # Kirim vektor nol sesuai dimensi Qwen2.5 (2048)
                embeddings.append([0.0] * 2048)
        return embeddings

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Metode yang dicari ChromaDB saat menjalankan .query()"""
        # Untuk Ollama, logika embed query sama dengan dokumen
        return self.embed_documents(input)
# --- LOGIKA RAG ENGINE ---
class RAGChatbot:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embed_fn = OllamaEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, 
            embedding_function=self.embed_fn
        )

    def process_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        # Chunking Logic (Fixed operator << to <)
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        chunk_size = 800 
        
        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Bersihkan chunk kosong/pendek
        chunks = [c for c in chunks if len(c) > 15]
        
        if chunks:
            # Tambahkan ID unik agar tidak konflik saat upload ulang
            ids = [f"doc_{os.urandom(4).hex()}_{i}" for i in range(len(chunks))]
            self.collection.add(documents=chunks, ids=ids)
            return len(chunks)
        return 0

    def ask(self, query, language="English"):
        print(f"🔍 Mencari konteks untuk: {query} (Language: {language})")
        
        # Query ke Vector DB
        results = self.collection.query(query_texts=[query], n_results=5)
        
        if not results["documents"] or not results["documents"][0]:
            no_info_msgs = {
                "Indonesia": "Maaf, tidak ada informasi relevan dalam dokumen yang diunggah.",
                "English": "Sorry, there is no relevant information in the uploaded document.",
                "Mandarin": "对不起，上传的文件中没有相关信息。"
            }
            return no_info_msgs.get(language, no_info_msgs["English"])
            
        context = "\n---\n".join(results["documents"][0])
        
        # Mapping bahasa ke instruksi prompt
        lang_map = {
            "Indonesia": "bahasa Indonesia",
            "English": "English",
            "Mandarin": "Mandarin (Chinese)"
        }
        target_lang = lang_map.get(language, "English")

        # Prompt Engineering untuk Sistem Pakar
        system_prompt = (
            f"Anda adalah Sistem Pakar yang disiplin. Gunakan konteks berikut untuk menjawab.\n\n"
            f"KONTEKS:\n{context}\n\n"
            f"PERTANYAAN: {query}\n\n"
            f"ATURAN: Jika jawaban tidak ada di konteks, katakan Anda tidak tahu. Jawablah dengan {target_lang} yang baik. JANGAN MENJAWAB DENGAN KATA KATA YANG TIDAK RELEVAN\n\n"
            f"JAWABAN:"
        )
        
        try:
            # Memanggil model lokal
            response = client.generate(model=OLLAMA_MODEL, prompt=system_prompt)
            return response['response']
        except Exception as e:
            return f"❌ Error LLM: {str(e)}"

# Global Instance
bot = RAGChatbot()

# --- API ENDPOINTS ---

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Simpan file sementara
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        num_chunks = bot.process_pdf(temp_path)
        os.remove(temp_path) # Hapus file temp
        
        return {"status": "success", "message": f"Berhasil memproses {num_chunks} potongan teks."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(data: dict):
    query = data.get("query")
    language = data.get("language", "English")
    if not query:
        return {"answer": "Silakan masukkan pertanyaan."}
    
    answer = bot.ask(query, language=language)
    return {"answer": answer}

@app.post("/reset")
async def reset_db():
    try:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
        # Re-inisialisasi bot agar database segar
        global bot
        bot = RAGChatbot()
        return {"status": "success", "message": "Database berhasil dikosongkan."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Jalankan di port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)