# 🤖 RAG Expert Chatbot

Sistem Pakar berbasis **Retrieval-Augmented Generation (RAG)** yang memungkinkan pengguna untuk berinteraksi dengan dokumen PDF. Chatbot ini dirancang untuk memberikan jawaban yang akurat dan terkontrol hanya berdasarkan isi dokumen yang diunggah, guna meminimalisir halusinasi AI.

## 📌 Overview
Proyek ini dibangun untuk memenuhi tugas mata kuliah **Expert System**. Fokus utama dari sistem ini adalah implementasi pipeline RAG yang menggabungkan keunggulan *Cloud Embedding* (untuk akurasi pencarian) dan *Local LLM* (untuk privasi dan efisiensi biaya).

### Alur Kerja Sistem:
1. **Indexing**: Dokumen PDF diunggah $\rightarrow$ Teks diekstrak $\rightarrow$ Teks dipecah menjadi *chunks* $\rightarrow$ Diubah menjadi vektor menggunakan **Gemini Embedding API** $\rightarrow$ Disimpan di **ChromaDB**.
2. **Retrieval**: Pertanyaan user diubah menjadi vektor $\rightarrow$ Sistem mencari potongan teks paling relevan di ChromaDB.
3. **Generation**: Potongan teks relevan (konteks) + pertanyaan user dikirim ke **Ollama (Local LLM)** dengan instruksi ketat untuk hanya menjawab berdasarkan konteks tersebut.

---

## 🛠️ Tech Stack
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Frontend**: HTML5 & CSS3 (Custom UI)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding Model**: Google Gemini `text-embedding-004` (Cloud API)
- **Large Language Model (LLM)**: [Ollama](https://ollama.com/) (Local Model: `gemma2` / `llama3`)
- **PDF Parsing**: PyPDF2

---

## 🚀 Instalasi & Penggunaan

### 1. Prasyarat
- Python 3.10+
- Ollama terinstall di lokal
- API Key dari [Google AI Studio](https://aistudio.google.com/)

### 2. Setup Environment
Kloning folder project atau masuk ke direktori project, lalu install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Model Lokal
Download model yang ingin digunakan melalui Ollama:
```bash
ollama pull gemma2
```

### 4. Konfigurasi API Key
Buka file `main.py` dan masukkan API Key Gemini kamu pada variabel:
```python
GOOGLE_API_KEY = "ISI_API_KEY_GEMINI_KAMU"
```

### 5. Menjalankan Aplikasi
Jalankan server FastAPI:
```bash
python main.py
```
Buka browser dan akses: `http://localhost:8000`

---

## 📁 Struktur Folder
```text
FINALEPROJECT/
├── static/
│   └── index.html      # Tampilan antarmuka chat (Frontend)
├── chroma_db/           # Penyimpanan vektor dokumen (Otomatis)
├── main.py             # Logika Backend, API, dan RAG Pipeline
└── requirements.txt     # Daftar library Python yang dibutuhkan
```

## 🎓 Catatan Akademik (Expert System)
Sistem ini menerapkan **Strict Context Prompting**. Berbeda dengan chatbot umum, sistem ini tidak menggunakan pengetahuan umum LLM jika informasi tidak tersedia di dokumen. Hal ini memastikan bahwa sistem berperan sebagai *Expert* pada domain dokumen tertentu, yang merupakan karakteristik utama dari sebuah Sistem Pakar.
