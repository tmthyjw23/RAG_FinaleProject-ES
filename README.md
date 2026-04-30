# 🚀 G4 Expert System: Multi-Tenant RAG Chatbot

Sistem Pakar berbasis **Retrieval-Augmented Generation (RAG)** modern yang dirancang untuk lingkungan *Multi-Tenant*. Chatbot ini mengizinkan pengguna untuk berinteraksi dengan dokumen PDF dalam ruang kerja (sesi) yang terisolasi. Dibangun dengan fleksibilitas tinggi, sistem ini mendukung mode komputasi hibrida yang memungkinkan pengguna memilih antara pemrosesan AI lokal berorientasi privasi atau kekuatan *Cloud GenAI*.

## 📌 Overview
Proyek ini dikembangkan sebagai implementasi tingkat lanjut dari Sistem Pakar. Dengan menerapkan **Strict Context Prompting** dan **Session-based Metadata Filtering**, sistem menjamin bahwa respons AI hanya berasal dari dokumen relevan yang diunggah oleh pengguna di sesi aktifnya, mencegah kebocoran data antar pengguna (*data bleed*) dan meminimalisir halusinasi AI.

### ✨ Fitur Utama
*   🔐 **Multi-Tenant Document Isolation:** Setiap pengunjung secara otomatis mendapatkan *Session ID* unik. Dokumen yang diunggah dan vektor yang di-*embed* disegel dengan *metadata* sesi, memastikan privasi absolut.
*   🔀 **Hybrid LLM Routing:** Tersedia *toggle* mode layanan:
    *   **G4 Local:** Menggunakan model lokal (Ollama) yang berjalan langsung di VPS untuk privasi 100%. Dilindungi oleh *Master Password*.
    *   **Cloud BYOK (Bring Your Own Key):** Menggunakan *library* OpenAI-compatible untuk menghubungkan API Key pribadi pengguna ke layanan eksternal (Groq, OpenRouter, Gemini, dll).
*   🧠 **Contextual Chat Memory:** Sistem RAG tidak hanya membaca dokumen, tetapi juga "mengingat" 4 interaksi terakhir untuk memberikan jawaban yang berkesinambungan (*follow-up questions*).
*   📄 **Granular File Management:** Kemampuan untuk menghapus dokumen spesifik (PDF) secara aman dari *Vector Database* tanpa merusak dokumen lain di sesi yang sama.
*   🛡️ **API Security:** Perlindungan *endpoint* menggunakan autentikasi *Bearer Token* (`HTTPBearer`).

---

## 🛠️ Tech Stack

### Backend & AI Pipeline
*   **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local Persistent)
*   **Local LLM Engine:** [Ollama](https://ollama.com/) (Model: `qwen2.5:0.5b`)
*   **Cloud Gateway:** [OpenAI Python SDK](https://github.com/openai/openai-python)
*   **Document Parsing:** PyPDF2

### Frontend & UI
*   **Library:** React.js 18 (Standalone via Babel)
*   **Styling:** Custom CSS3 (Modern Glassmorphism, Responsive)
*   **State Management:** LocalStorage & SessionStorage untuk persistensi sesi.

### Production Environment
*   **OS:** OpenCloudOS (Linux VPS)
*   **Process Manager:** `systemd` (Uvicorn Workers)
*   **Tunneling/Proxy:** Cloudflare Tunnels (Zero Trust Network Access)

---

## ⚙️ Alur Kerja Sistem (Pipeline)

1. **Ingestion & Isolation**: PDF Diunggah -> Teks diekstraksi & di-*chunk* -> Di-*embed* menggunakan Ollama -> Disimpan ke ChromaDB dengan sisipan Metadata `{"session_id": "uuid", "filename": "doc.pdf"}`.
2. **Context Retrieval**: Pertanyaan User -> Filter pencarian ChromaDB membatasi target hanya pada dokumen dengan `session_id` yang sesuai -> Potongan teks relevan diekstraksi.
3. **Prompt Construction**: Konteks PDF + Riwayat Obrolan (Memori) + Pertanyaan Baru digabungkan menjadi satu *System Prompt* yang ketat.
4. **Generation (Dynamic Routing)**: Berdasarkan mode yang dipilih, prompt dikirim ke lokal (Ollama) atau dikirim ke Cloud Gateway menggunakan *API Key* pengguna.

---

## 🚀 Panduan Instalasi & Penggunaan Lokal

### 1. Prasyarat
*   Python 3.10+
*   [Ollama](https://ollama.com/download) terinstal di sistem operasi Anda.

### 2. Setup Environment
Kloning repositori ini, lalu masuk ke direktori proyek dan instal semua *dependencies*:
```bash
git clone [https://github.com/tmthyjw23/RAG_FinaleProject-ES.git](https://github.com/tmthyjw23/RAG_FinaleProject-ES.git
cd G4-Expert-System
pip install -r requirements.txt