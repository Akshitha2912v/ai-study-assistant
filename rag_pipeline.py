import os
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv

load_dotenv()

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready!")

# ── STEP 1: Read the PDF ───────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1
            })
    return pages

# ── STEP 2: Split into chunks ──────────────────────────────────────────────────
def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num
                })
            start = end - overlap
    return chunks

# ── STEP 3: Convert chunks to vectors ─────────────────────────────────────────
def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# ── STEP 4: Find relevant chunks ──────────────────────────────────────────────
def retrieve(question, index, chunks, top_k=4):
    question_vector = embedder.encode([question]).astype("float32")
    distances, indices = index.search(question_vector, top_k)
    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])
    return results

# ── STEP 5: Generate answer using Ollama (free, runs on your Mac) ──────────────
def generate_answer(question, relevant_chunks):
    context = ""
    sources = []
    for chunk in relevant_chunks:
        context += f"\n[Page {chunk['page']}]\n{chunk['text']}\n"
        if chunk["page"] not in sources:
            sources.append(chunk["page"])

    prompt = f"""You are a helpful study assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]
    return answer, sources

# ── MASTER FUNCTION ────────────────────────────────────────────────────────────
def process_pdf_and_answer(pdf_path, question):
    print(f"\n--- Processing: {pdf_path} ---")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages")

    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")

    index, chunks = build_index(chunks)
    print(f"Built search index")

    relevant_chunks = retrieve(question, index, chunks)
    print(f"Found {len(relevant_chunks)} relevant chunks")

    answer, sources = generate_answer(question, relevant_chunks)
    return answer, sources