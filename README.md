#  Voice-Enabled RAG Chatbot

An end-to-end **voice-enabled Retrieval-Augmented Generation (RAG) system** that answers user questions from Hindi speech input by combining **ASR, Translation, Vector Search, and LLM-based Answer Generation**.

---

##  Overview

This project implements all **five tasks**:

1. **Wikipedia data collection**
2. **Vector database creation**
3. **ASR deployment using FastAPI**
4. **Hindi → English translation**
5. **End-to-end RAG pipeline with a UI**

A **Gradio-based web interface** allows users to upload a Hindi `.wav` audio file and receive an AI-generated answer.

---

## System Architecture

Audio (.wav)  
↓  
ASR (IndicConformer_Hi)  
↓  
Hindi → English Translation (Sarvam API)  
↓  
Vector Retrieval (FAISS)  
↓  
LLM Answer Generation (GitHub Models)  
↓  
Final Answer (UI)

---

##  Repository Structure
```
RAG-voice-chatbot/
│
├── data/
│ └── wikipedia.txt
│
├── vector_db/
│ └── faiss_index/
│
├── models/
│ └── indicconformer_stt_hi_hybrid_rnnt_large.nemo
│
├── scripts/
│ ├── app.py # Gradio UI
│ ├── task1_wiki_scrape.py # Wikipedia scraping
│ ├── task2_vector_db.py # Vector DB creation
│ ├── task3_asr.py # ASR FastAPI service
│ ├── task4_translate.py # Translation API
│ └── task5_rag_pipeline.py # End-to-end RAG pipeline
│
├── Notebook/
│ └── RAG_Notebook.ipynb # Development notebook
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

##  Task Details

###  Task-1: Data Collection
- Accepts a **topic as a command-line argument**
- Uses **Google Search (SerpAPI)** to find the closest Wikipedia article
- Scrapes clean article text using `requests` + `BeautifulSoup`
- Stores the extracted content in a `.txt` file

---

###  Task-2: Vector Database Creation

**Chunking Strategy**
- Chunk size: **~500 characters**
- Overlap: **100 characters**
- Reason: preserves semantic continuity across chunk boundaries

**Embedding**
- Model: `all-MiniLM-L6-v2`
- Embedding dimension: **384**
- Total chunks: **258**

**Vector Store**
- Database: **FAISS**
- Chosen for:
  - Fast similarity search
  - Lightweight and open-source
  - Easy local persistence

---

###  Task-3: ASR Deployment (FastAPI)

- Model: **IndicConformer (Hindi)** from NeMo
- Framework: **FastAPI**
- Endpoint:
  - `POST /transcribe` → accepts `.wav` audio
  - Returns Hindi transcription
- Health check:
  - `GET /` → service status

---

###  Task-4: Translation

- Uses **Sarvam Translation API**
- Converts Hindi text → English
- Simple API call (no deployment)

---

###  Task-5: Final RAG Pipeline

Steps:
1. Audio input
2. ASR transcription
3. Hindi → English translation
4. Retrieve **top-2** relevant chunks from FAISS
5. Generate answer using **GitHub Models**

**LLM Used**
- Provider: GitHub Models  
- Model: `gpt-4o-mini`

---

##  Bonus: Gradio UI

- Upload Hindi `.wav` audio
- Click **Ask Question**
- Displays AI-generated answer
- Public shareable link generated via Gradio

 **UI Screenshot**  
![UI](Voice-enabled%20AI%20chatbot%20interface.png)

---

##  Setup Instructions

```bash
git clone https://github.com/Kahkashan2708/RAG-voice-chatbot.git
cd RAG-voice-chatbot
pip install -r requirements.txt
```
---


## Set environment variables:
```bash
* export GITHUB_TOKEN=your_token_here
* export SARVAM_API_KEY=your_key_here
```

