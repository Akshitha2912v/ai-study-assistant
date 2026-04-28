# 📚 AI Study Assistant (RAG-based)

## 🚀 Overview
An AI-powered study assistant that processes PDF documents and generates answers using Retrieval-Augmented Generation (RAG).  
It extracts content from documents, converts them into embeddings, retrieves relevant information, and generates meaningful answers.

---

## ✨ Features
- 📄 PDF text extraction  
- 🧩 Text chunking  
- 🔍 Semantic search using FAISS  
- 🤖 Answer generation using LLM (Ollama/OpenAI)  
- ⚡ Fast and efficient retrieval system  

---

## 🧠 Tech Stack
- Python  
- Sentence Transformers  
- FAISS (Vector Search)  
- PyMuPDF / pdfplumber  
- Ollama / OpenAI  
- RAG (Retrieval-Augmented Generation)

---

## 📂 Project Structure
ai-study-assistant/
│── rag_pipeline.py        # Core RAG logic
│── test.py                # Run and test the system
│── uploads/               # Input PDFs
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
