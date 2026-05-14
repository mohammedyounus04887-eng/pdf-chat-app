from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Gemini Client (NEW SDK)
# =========================
client = genai.Client(api_key="AIzaSyCwkE4VhsBmk7LXGBDDJL8R0pjrvKMN5l4")

# Correct model name (safe & working)
GEMINI_MODEL = "gemini-flash-latest"

# =========================
# Embedding model
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Global storage
# =========================
pdf_chunks = []
vector_store = None


@app.get("/")
def home():
    return {"message": "Backend running successfully 🚀"}


# =========================
# Upload PDF
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks, vector_store

    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        # Chunk text
        chunk_size = 500
        pdf_chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

        # Embeddings
        embeddings = embedding_model.encode(pdf_chunks)
        embeddings = np.array(embeddings).astype("float32")

        # FAISS index
        dimension = embeddings.shape[1]
        vector_store = faiss.IndexFlatL2(dimension)
        vector_store.add(embeddings)

        return {
            "message": "PDF uploaded successfully",
            "chunks": len(pdf_chunks)
        }

    except Exception as e:
        return {"detail": str(e)}


# =========================
# Ask Question
# =========================
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global pdf_chunks, vector_store

    try:
        if vector_store is None:
            return {"detail": "Upload a PDF first"}

        # Embed question
        q_embedding = embedding_model.encode([question])
        q_embedding = np.array(q_embedding).astype("float32")

        # Search
        k = 3
        distances, indices = vector_store.search(q_embedding, k)

        context = "\n".join([pdf_chunks[i] for i in indices[0]])

        prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

        # =========================
        # CORRECT GEMINI CALL
        # =========================
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return {
            "answer": response.text
        }

    except Exception as e:
        return {"detail": str(e)}