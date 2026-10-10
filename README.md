# Technical Test — AI Specialist Intern

### PT Link Apisindo Media · Gregorius Olvans

A dual-deliverable AI engineering submission combining a **Retrieval-Augmented Generation (RAG) API** for financial-report question answering and a **Computer Vision OCR pipeline** that reconstructs slide images into interactive, editable HTML.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-0.5-FF6F00?logo=databricks&logoColor=white">
  <img alt="Gemini" src="https://img.shields.io/badge/Gemini-3.8--flash-4285F4?logo=google&logoColor=white">
  <img alt="EasyOCR" src="https://img.shields.io/badge/EasyOCR-1.7-EE4C2C?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 🎬 Video Walkthrough Demo

> **Full explanation & live demonstration of both tasks:**
>
> ### ▶️ **[Watch the Walkthrough on Google Drive](https://drive.google.com/drive/folders/1pl4C2d1_tVcdMh_46o8z6g9B8YkGbpos?usp=sharing)**

---

## 📦 Repository Structure

```
.
├── README.md                        # This documentation
├── .gitignore                       # Secret & artifact protection
├── TaskA_RAG_LaporanBank/           # Task A — RAG API
│   ├── app.py                       # FastAPI RAG service
│   ├── requirements.txt
│   ├── .env.example                 # Environment template
│   ├── Laporan Keuangan Bank Mandiri 2025.pdf
│   └── chroma_db/                   # Persisted vector store
└── TaskB_OCR_HTML/                  # Task B — CV / OCR pipeline
    ├── extract_to_html.py           # OCR → interactive HTML
    ├── requirements.txt
    ├── slide.jpg                    # Source slide image
    └── output.html                  # Generated editable HTML
```

---

## 🧠 Task A — RAG API for Bank Mandiri Financial Reports

A production-style **Retrieval-Augmented Generation API** that ingests the *Laporan Keuangan Bank Mandiri 2025* PDF and answers natural-language questions grounded strictly in the document.

### Architecture

```
 PDF Upload ─▶ PyMuPDFLoader ─▶ RecursiveCharacterTextSplitter ─▶ HuggingFace Embeddings
                                                                        │
                                                                        ▼
   HTTP GET /ask ─▶ Similarity Search (k=3) ◀──────────────  ChromaDB (persisted)
                            │
                            ▼
              PromptTemplate ─▶ Gemini 3.8-flash (temp=0) ─▶ Grounded Answer
```

### Key Design Decisions

| Component | Technology | Purpose |
|---|---|---|
| **API framework** | FastAPI + Uvicorn | Async, typed, auto-generated Swagger docs |
| **PDF parsing** | PyMuPDF (`PyMuPDFLoader`) | Fast, stable text & table extraction |
| **Chunking** | `RecursiveCharacterTextSplitter` | `chunk_size=1000`, `chunk_overlap=200` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local, cost-free semantic vectors |
| **Vector store** | ChromaDB (persisted to `./chroma_db`) | Durable similarity search |
| **LLM** | `gemini-3.8-flash` | High-capability reasoning |
| **Temperature** | **`0`** | **Eliminates hallucination — deterministic output** |

### Anti-Hallucination Strategy

- **`temperature=0`** forces the most probable, deterministic token path — no creative drift.
- **Strict grounding prompt**: the model is instructed to answer *only* from retrieved context and to explicitly state when information is unavailable.
- **`k=3` retrieval** keeps context tight and relevant, reducing noise that can mislead the model.
- Deterministic extraction of **credit/loan tables** and **DPK (Dana Pihak Ketiga / third-party funds) ratios** with reproducible results.
- **Out-of-domain guardrail**: queries unrelated to the report are refused rather than answered speculatively.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF; parses, chunks, embeds, and persists to ChromaDB |
| `GET` | `/ask?query=...` | Semantic retrieval + LLM answer grounded in the document |

---

## 👁️ Task B — Computer Vision OCR → Interactive HTML

A pipeline that reads a presentation slide image and rebuilds every text block as an **absolutely-positioned, editable HTML element** overlaid on the original image — producing a pixel-aligned, editable reconstruction of the document.

### Pipeline

```
 slide.jpg ─▶ OpenCV (read) ─▶ EasyOCR (detect + recognize, id/en)
                                        │
                                        ▼
                     Per-block bounding box → background sampling (median color)
                                        │
                                        ▼
          ITU-R BT.601 luminance → adaptive text color (contrast guard)
                                        │
                                        ▼
        Absolute DOM reconstruction with contenteditable="true"
                                        │
                                        ▼
                               output.html (interactive)
```

### Technical Highlights

- **OCR engine**: `EasyOCR` configured for **Indonesian + English** (`['id', 'en']`).
- **Adaptive text color contrast**: each block's background is sampled via **median color** over a padded region, then converted with the **ITU-R BT.601 luminance formula**:
  ```
  Y = 0.299·R + 0.587·G + 0.114·B
  ```
  Text is set to **black** when `Y > 128`, otherwise **white** — guaranteeing legibility regardless of background.
- **Dynamic absolute DOM reconstruction**: every detected block becomes a `<span class="text-box">` positioned with the exact `left/top/width/height` of its bounding box, with `font-size` scaled to box height.
- **`contenteditable="true"`**: every text element is directly editable in the browser, with hover/focus highlighting and a confidence tooltip.

### Output

Open `output.html` in any browser to view the slide with fully editable, correctly-contrasted text overlaid on the source image.

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python 3.10+**
- **pip** and **virtualenv**
- A **Google Gemini API key** ([Google AI Studio](https://aistudio.google.com/app/apikey))

### 1. Clone the repository

```bash
git clone https://github.com/Itsmevnztxt/Technical-TestAI-Specialist-Intern-Gregorius-Olvans-PT-Link-Apisindo-Media.git
cd Technical-TestAI-Specialist-Intern-Gregorius-Olvans-PT-Link-Apisindo-Media
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Task A — RAG API
pip install -r TaskA_RAG_LaporanBank/requirements.txt

# Task B — OCR pipeline
pip install -r TaskB_OCR_HTML/requirements.txt
```

### 4. Configure environment variables

```powershell
copy TaskA_RAG_LaporanBank\.env.example TaskA_RAG_LaporanBank\.env
```

Then edit `TaskA_RAG_LaporanBank/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

> ⚠️ **Never commit the real `.env`.** It is protected by `.gitignore`.

---

## ▶️ Running the Project

### Task A — Start the RAG API server

```bash
cd TaskA_RAG_LaporanBank
uvicorn app:app --reload
```

The service starts at `http://localhost:8000`. Interactive Swagger docs: `http://localhost:8000/docs`.

**Example usage:**

```bash
# 1) Upload the financial report
curl -X POST "http://localhost:8000/upload" \
     -F "file=@Laporan Keuangan Bank Mandiri 2025.pdf"

# 2) Ask a grounded question
curl "http://localhost:8000/ask?query=Berapa rasio DPK Bank Mandiri 2025?"
```

### Task B — Generate the interactive HTML

```bash
cd TaskB_OCR_HTML
python extract_to_html.py
```

Optionally pass a custom image path:

```bash
python extract_to_html.py "path/to/your/slide.jpg"
```

Open the generated **`output.html`** in your browser and click any text block to edit it.

---

## 🔐 Security Notes

- `.env` (containing the real API key) is **git-ignored** and never committed.
- `.env.example` ships with placeholder values only.
- Runtime uploads and Python build artifacts (`__pycache__/`, `.venv/`, temp files) are excluded.

---

## 📄 License

Released under the **MIT License** — see repository for details.

<p align="center"><i>Built with FastAPI, LangChain, ChromaDB, OpenCV, and EasyOCR.</i></p>
