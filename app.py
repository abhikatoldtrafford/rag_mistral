import time
import threading
import uuid
import os
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
import csv

import uvicorn
import torch
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

import logging

# -------------------------------------------------------------------
# 1. Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 2. FastAPI App Initialization
# -------------------------------------------------------------------
app = FastAPI(
    title="RAG API Service",
    description=(
        "A retrieval-augmented generation service with conversational memory. "
        "The file index represents a GitHub repository (with preserved file paths), and chat history is maintained "
        "to support continuous conversation."
    ),
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------------------------------------------
# 3. Directory & Constants Setup
# -------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
INDEX_DIR = Path("faiss_indexes")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 128

# Recognized text-based extensions (code, markdown, HTML, etc.)
TEXT_BASED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".html", ".htm", ".css", ".c", ".cpp", ".cc", ".h", ".hpp",
    ".json", ".md", ".txt", ".xml", ".yaml", ".yml", ".go", ".rs", ".sh", ".php", ".rb", ".lua",
    ".cs", ".dart"
}

# A simple CSV file to store user->index->status
USER_DATA_CSV = Path("users.csv")

# -------------------------------------------------------------------
# 4. Model Setup
# -------------------------------------------------------------------
model_name = "mistralai/Ministral-8B-Instruct-2410"
logger.info("Loading LLM...")

llm = LLM(
    model=model_name,
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
    device='cuda',
    dtype=torch.bfloat16,
    max_model_len=10000,
    enforce_eager=True
)
llm_lock = threading.Lock()

# -------------------------------------------------------------------
# 5. Embedding Model
# -------------------------------------------------------------------
logger.info("Loading embedding model...")
embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
embedding_lock = threading.Lock()

# -------------------------------------------------------------------
# 6. Pydantic Models
# -------------------------------------------------------------------
class RequestBody(BaseModel):
    inputs: str

class RagRequestBody(BaseModel):
    query: str
    index_id: Optional[str] = None

class ChatRequestBody(BaseModel):
    query: str
    index_id: str

class FileUploadResponse(BaseModel):
    index_id: str
    num_documents: int
    message: str

class ClearChatRequest(BaseModel):
    index_id: str

class InitiateChatRequest(BaseModel):
    user_id: str

class InitiateChatResponse(BaseModel):
    user_id: str
    index_id: str
    message: str

class UploadStatusResponse(BaseModel):
    status: int  # -1 = indexing in progress, 0 = indexing complete
    message: str

# -------------------------------------------------------------------
# 7. CSV Utility Functions
# -------------------------------------------------------------------
def ensure_csv_exists():
    """Create the users.csv if not already present, with a header row."""
    if not USER_DATA_CSV.exists():
        with open(USER_DATA_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "index_id", "status"])  # status = -1 or 0

def read_user_data() -> List[dict]:
    """Read the user data from CSV into a list of dicts."""
    ensure_csv_exists()
    rows = []
    with open(USER_DATA_CSV, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def write_user_data(rows: List[dict]):
    """Write the user data dicts back to CSV."""
    ensure_csv_exists()
    with open(USER_DATA_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "index_id", "status"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def find_user(user_id: str) -> Optional[dict]:
    """Find a user by user_id in the CSV."""
    data = read_user_data()
    for row in data:
        if row["user_id"] == user_id:
            return row
    return None

def update_user(user_id: str, index_id: str, status: str):
    """Update or insert user data in the CSV."""
    data = read_user_data()
    found = False
    for row in data:
        if row["user_id"] == user_id:
            row["index_id"] = index_id
            row["status"] = status
            found = True
            break
    if not found:
        data.append({"user_id": user_id, "index_id": index_id, "status": status})
    write_user_data(data)

def get_status_by_index_id(index_id: str) -> int:
    """Return the status (int) for a given index_id from CSV or raise if not found."""
    data = read_user_data()
    for row in data:
        if row["index_id"] == index_id:
            return int(row["status"])
    # If not found, default to 0 or raise error
    raise HTTPException(status_code=404, detail=f"Index {index_id} not found in CSV.")

def set_status_by_index_id(index_id: str, status: int):
    """Set the status (int) for a given index_id."""
    data = read_user_data()
    updated = False
    for row in data:
        if row["index_id"] == index_id:
            row["status"] = str(status)
            updated = True
            break
    if updated:
        write_user_data(data)
    else:
        raise HTTPException(status_code=404, detail=f"Index {index_id} not found in CSV.")

# -------------------------------------------------------------------
# 8. Embedding & Index Utility
# -------------------------------------------------------------------
def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using the shared embedding model."""
    with embedding_lock:
        return embedding_model.encode(texts, convert_to_numpy=True)

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def unzip_and_collect_documents(zip_path: Path) -> List[dict]:
    """
    Extract a ZIP file from a saved path while preserving folder structure.
    Returns a list of document chunks with metadata:
      - "text": the chunk content
      - "source_file": the relative file path
      - "chunk_size": number of words in the chunk
    """
    temp_folder = Path(zip_path).parent  # the folder containing "uploaded.zip"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_folder)

    documents = []
    for root, _, files in os.walk(temp_folder):
        # skip "uploaded.zip" if present
        if root == str(temp_folder) and "uploaded.zip" in files:
            files.remove("uploaded.zip")
        for filename in files:
            file_path = Path(root) / filename
            rel_path = file_path.relative_to(temp_folder)
            if file_path.suffix.lower() not in TEXT_BASED_EXTENSIONS:
                logger.info(f"Skipping non-text file: {rel_path}")
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_content = f.read()
                for chunk in chunk_text(raw_content):
                    documents.append({
                        "text": chunk,
                        "source_file": str(rel_path),
                        "chunk_size": len(chunk.split())
                    })
            except Exception as e:
                logger.warning(f"Error reading {rel_path}: {e}")
                continue

    return documents

def create_faiss_index(documents: List[dict]) -> Tuple[np.ndarray, faiss.IndexFlatL2]:
    """
    Create a FAISS index from document chunks.
    """
    texts = [doc["text"] for doc in documents]
    embeddings = embed_batch(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return embeddings, index

def save_index_and_metadata(index_id: str, index: faiss.IndexFlatL2, documents: List[dict], embeddings: np.ndarray):
    """
    Save the FAISS index, embeddings, and document metadata.
    """
    index_path = INDEX_DIR / index_id
    index_path.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_path / "index.faiss"))
    np.save(str(index_path / "embeddings.npy"), embeddings)
    with open(index_path / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)

def load_index_and_metadata(index_id: str) -> Tuple[faiss.IndexFlatL2, List[dict], np.ndarray]:
    """
    Load the FAISS index, embeddings, and document metadata.
    """
    index_path = INDEX_DIR / index_id
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Index '{index_id}' not found.")
    faiss_index_file = index_path / "index.faiss"
    embeddings_file = index_path / "embeddings.npy"
    documents_file = index_path / "documents.json"
    if not (faiss_index_file.is_file() and embeddings_file.is_file() and documents_file.is_file()):
        raise HTTPException(status_code=404, detail=f"Incomplete index data for '{index_id}'.")
    index = faiss.read_index(str(faiss_index_file))
    embeddings = np.load(str(embeddings_file))
    with open(documents_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    return index, documents, embeddings

# -------------------------------------------------------------------
# 9. Index Lock Management
# -------------------------------------------------------------------
index_locks = {}
global_lock = threading.Lock()

def get_index_lock(index_id: str) -> threading.Lock:
    """Return a per-index lock for thread-safe updates."""
    with global_lock:
        if index_id not in index_locks:
            index_locks[index_id] = threading.Lock()
        return index_locks[index_id]

# -------------------------------------------------------------------
# 10. Initiate Chat
# -------------------------------------------------------------------
@app.post("/initiate_chat/", response_model=InitiateChatResponse)
def initiate_chat(request: InitiateChatRequest):
    """
    Basic "login" or session start. 
    - If user_id exists, returns existing index_id. 
    - Otherwise, creates a new index_id for that user.
    """
    user_id = request.user_id
    row = find_user(user_id)
    if row:  # existing user
        message = f"Returning existing index_id for user_id '{user_id}'."
        return InitiateChatResponse(
            user_id=user_id,
            index_id=row["index_id"],
            message=message
        )
    else:
        # Create a new index_id
        new_index_id = str(uuid.uuid4())
        # By default, set status = 0 (meaning no indexing yet, or "idle")
        update_user(user_id, new_index_id, status="0")
        message = f"Created new index_id for user_id '{user_id}'."
        return InitiateChatResponse(
            user_id=user_id,
            index_id=new_index_id,
            message=message
        )

# -------------------------------------------------------------------
# 11. Background Task for Indexing
# -------------------------------------------------------------------
def do_indexing(index_id: str, zip_file_path: Path):
    """
    The background task that unzips and indexes documents, then updates status.
    """
    try:
        lock = get_index_lock(index_id)
        with lock:
            documents = unzip_and_collect_documents(zip_file_path)
            if not documents:
                # If no valid docs, revert status to 0
                set_status_by_index_id(index_id, 0)
                return

            embeddings, faiss_index = create_faiss_index(documents)
            save_index_and_metadata(index_id, faiss_index, documents, embeddings)
            logger.info(f"Background indexing completed for index_id={index_id}.")
            # Mark status = 0 (done)
            set_status_by_index_id(index_id, 0)
    except Exception as e:
        logger.error(f"Background indexing error for index_id={index_id}: {e}", exc_info=True)
        set_status_by_index_id(index_id, 0)  # Reset to 0 on failure

# -------------------------------------------------------------------
# 12. Async Endpoint: Upload File (Background Task)
# -------------------------------------------------------------------
@app.post("/upload_file_async/", response_model=UploadStatusResponse)
def upload_file_async(
    index_id: str,
    background_tasks: BackgroundTasks,
    zipfile: UploadFile = File(None)
):
    """
    - If no file is provided, just return the current status for that index_id.
    - If a file is provided, store it to disk & schedule a background task to do the indexing.
      => Immediately return: { status: -1, message: "Indexing started" }
    """
    if not index_id:
        raise HTTPException(status_code=400, detail="index_id is required.")

    # If no file => status check only
    if zipfile is None:
        current_status = get_status_by_index_id(index_id)
        msg = "Indexing is still in progress." if current_status == -1 else "Indexing is complete."
        return UploadStatusResponse(status=current_status, message=msg)

    # If file => start background indexing
    # 1) Set status = -1
    set_status_by_index_id(index_id, -1)
    
    # 2) Save the uploaded file to a temp folder
    temp_folder = UPLOAD_DIR / str(uuid.uuid4())
    temp_folder.mkdir(parents=True, exist_ok=True)
    zip_path = temp_folder / "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(zipfile.file.read())

    # 3) Schedule background task for indexing
    background_tasks.add_task(do_indexing, index_id, zip_path)
    logger.info(f"Background task scheduled for index_id={index_id}")

    # Return immediately
    return UploadStatusResponse(
        status=-1,
        message="Indexing started in background."
    )

# -------------------------------------------------------------------
# 13. Remainder of Endpoints (Predict, RAG, Chat, Clear Chat)
# -------------------------------------------------------------------
@app.post("/predict/")
async def predict(body: RequestBody):
    """
    Direct inference endpoint without retrieval.
    """
    try:
        start = time.time()
        messages = [{"role": "user", "content": body.inputs.strip()}]
        with llm_lock:
            outputs = llm.chat(messages, SamplingParams(temperature=0.6, top_p=0.9, max_tokens=5000))
        return {
            "response": outputs[0].outputs[0].text,
            "stats": {
                "time": round(time.time() - start, 2),
                "in_tokens": len(outputs[0].prompt_token_ids),
                "out_tokens": len(outputs[0].outputs[0].token_ids)
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_rag/")
async def predict_rag(body: RagRequestBody):
    """
    RAG endpoint using the file index.
    """
    try:
        if not body.index_id:
            raise HTTPException(status_code=400, detail="index_id required.")
        index, documents, _ = load_index_and_metadata(body.index_id)
        query_embed = embed_batch([body.query])[0].reshape(1, -1)
        _, indices = index.search(query_embed, 5)
        context = "\n".join(
            [f"File: {documents[i]['source_file']}\n{documents[i]['text']}" for i in indices[0]]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a codebase expert. The following context is derived from a file index representing a GitHub repository. "
                    "Each excerpt shows its original file path. Use this context to answer the question and reference sources appropriately."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {body.query}"
            }
        ]
        start = time.time()
        with llm_lock:
            outputs = llm.chat(messages, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=5000))
        return {
            "response": outputs[0].outputs[0].text.strip(),
            "sources": [documents[i]['source_file'] for i in indices[0]],
            "stats": {
                "time": round(time.time() - start, 2),
                "in_tokens": len(outputs[0].prompt_token_ids),
                "out_tokens": len(outputs[0].outputs[0].token_ids)
            }
        }
    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/")
async def chat_with_history(body: ChatRequestBody):
    """
    Chat endpoint with conversational memory. 
    Uses the same FAISS index with appended user/assistant messages.
    """
    try:
        lock = get_index_lock(body.index_id)
        with lock:
            index, documents, embeddings = load_index_and_metadata(body.index_id)
            
            # Retrieve top 5 chunks for overall context
            query_embed = embed_batch([body.query])[0].reshape(1, -1)
            _, idxs = index.search(query_embed, 5)
            
            # Separate retrieved documents into code context and chat history
            code_context_list = []
            convo_context_list = []
            for i in idxs[0]:
                doc = documents[i]
                if "type" in doc:
                    convo_context_list.append(f"{doc['source_file']}:\n{doc['text']}")
                else:
                    code_context_list.append(f"{doc['source_file']}:\n{doc['text']}")
            code_context = "\n".join(code_context_list)
            previous_convo_context = "\n".join(convo_context_list[:2])  # top 2 chat chunks
            
            # Build system prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the following context to answer the current query:\n"
                        "code context::::\n" + code_context + "\n"
                        "previous_convo_context :::\n" + previous_convo_context + "\n"
                        "Answer the question clearly while maintaining conversation continuity."
                    )
                },
                {
                    "role": "user",
                    "content": f"Current Question: {body.query}"
                }
            ]
            start = time.time()
            with llm_lock:
                outputs = llm.chat(messages, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=5000))
            response = outputs[0].outputs[0].text.strip()
            
            # Append user + assistant messages to the index
            timestamp = int(time.time())
            new_docs = [
                {
                    "text": body.query,
                    "source_file": f"chat/{timestamp}_user",
                    "chunk_size": len(body.query.split()),
                    "type": "user_input"
                },
                {
                    "text": response,
                    "source_file": f"chat/{timestamp}_assistant",
                    "chunk_size": len(response.split()),
                    "type": "assistant_response"
                }
            ]
            new_embeddings = embed_batch([body.query, response])
            index.add(new_embeddings)
            updated_docs = documents + new_docs
            updated_embeds = np.vstack([embeddings, new_embeddings])
            save_index_and_metadata(body.index_id, index, updated_docs, updated_embeds)
            
            return {
                "response": response,
                "context_sources": [documents[i]['source_file'] for i in idxs[0]],
                "stats": {
                    "time": round(time.time() - start, 2),
                    "in_tokens": len(outputs[0].prompt_token_ids),
                    "out_tokens": len(outputs[0].outputs[0].token_ids)
                }
            }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_chat/")
async def clear_chat(request: ClearChatRequest):
    """
    Clear the chat history for a given index by removing documents with a "type" field,
    then rebuild the FAISS index using only the original code-based documents.
    """
    try:
        lock = get_index_lock(request.index_id)
        with lock:
            index, documents, _ = load_index_and_metadata(request.index_id)
            # Filter out chat history documents (those with a "type" field)
            filtered_docs = [doc for doc in documents if "type" not in doc]
            new_embeddings, new_index = create_faiss_index(filtered_docs)
            save_index_and_metadata(request.index_id, new_index, filtered_docs, new_embeddings)
        return {"message": "Chat history cleared."}
    except Exception as e:
        logger.error(f"Clear chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# 14. Main Entry
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Note: use --workers to allow concurrency, e.g. `uvicorn main:app --workers 4`
    uvicorn.run(app, host="0.0.0.0", port=8000)
