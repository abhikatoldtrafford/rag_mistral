import time
import threading
import uuid
import os
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import csv
from datetime import datetime, timedelta
import re
from nltk.tokenize import sent_tokenize
from functools import lru_cache
import uvicorn
import torch
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

import logging, nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
system_prompt = '''
You are a specialized software development assistant with deep expertise in code analysis, troubleshooting, and test generation. Your primary focus should ALWAYS be addressing the user's immediate query with precision.

Priority Order for Information Sources:
1. User Query (HIGHEST PRIORITY)
   - Always address the specific question or task requested
   - If query is unclear, ask for clarification before proceeding
   - Never ignore any part of the user's query

2. Code Context (SECONDARY PRIORITY)
   - Provided in the format:
     File: <filename>
     <code or text content>
   - Reference this context if it's relevant to the query
   - Always cite specific files when using this context
   - If the context seems irrelevant with respect to user query, rely on general knowledge instead

3. Conversation History (TERTIARY PRIORITY)
   - Provided in the format:
     User: <previous question>
     Assistant: <previous response>
   - Use only to maintain consistency with previous interactions
   - Don't let historical context override the current query's needs

4. General Knowledge (FALLBACK)
   - Use when other sources don't provide relevant information
   - Clearly state when you're using general knowledge instead of provided context

Response Guidelines:
- Start responses by directly addressing the user's query
- Keep context references focused and relevant
- Format code sections with proper markdown
- When analyzing code:
  * Point out specific files and line references
  * Highlight potential issues
  * Suggest specific improvements
- For test scenarios:
  * Cover edge cases
  * Provide concrete examples
  * Structure tests logically

Remember: You will receive context in this specific structure:
1. PRIMARY: The user's query will be provided directly
2. CONTEXT: If available, code context will be clearly marked with "File:" headers
3. HISTORY: If available, conversation history will be formatted as User/Assistant pairs
4. SYSTEM: Your general knowledge serves as a fallback

Always validate that any context or history you reference is actually relevant to the current query before including it in your response.
'''
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
        "A retrieval-augmented generation service with short-term conversational memory. "
        "The file index represents a GitHub repository, and chat history is persisted separately "
        "for only the last hour."
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
THREADS_DIR = Path("threads")   # stores ephemeral conversation data

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
THREADS_DIR.mkdir(exist_ok=True)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 128
ONE_HOUR = 3600  # seconds
MAX_HISTORY_MESSAGES = 10  # max conversation turns to keep

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
    enforce_eager=True,
    gpu_memory_utilization=.8
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

class ChatRequestBody(BaseModel):
    query: str
    index_id: Optional[str] = None
    thread_id: Optional[str] = None

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

class ThreadCreateRequest(BaseModel):
    index_id: str

class ThreadResponse(BaseModel):
    thread_id: str
    created_at: float

class ChatResponse(BaseModel):
    response: str
    thread_id: Optional[str] = None
    sources: List[str] = []
    stats: Dict[str, float]



@lru_cache(maxsize=1000)
def cached_sent_tokenize(text: str) -> list:
    return sent_tokenize(text)

def is_list_item(text: str) -> bool:
    return text.strip().startswith(('- ', '* ', '• ', '1. ', '2. '))

def summarize_response(text: str, max_sentences: int = 3) -> str:
    """
    Summarizes a response by preserving code blocks, handling lists, and truncating to N sentences.
    """
    if not text or len(text.strip()) < 100:
        return text.strip()

    # Save code blocks
    code_blocks = []
    def save_code(match):
        code_blocks.append(match.group(0))
        return f"[CODE_BLOCK_{len(code_blocks)-1}]"

    cleaned = text.strip()
    cleaned = re.sub(r'(```[\s\S]*?```|`[^`]*`|^\s{4,}.*$)', save_code, cleaned, flags=re.MULTILINE)

    # Split into sentences
    sentences = cached_sent_tokenize(cleaned)
    summary = []
    list_in_progress = False

    for sentence in sentences:
        if is_list_item(sentence):
            list_in_progress = True
        elif list_in_progress and not sentence.strip():
            list_in_progress = False

        summary.append(sentence)
        if len(summary) >= max_sentences and not list_in_progress:
            break

    result = ' '.join(summary)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        result = result.replace(f"[CODE_BLOCK_{i}]", block)

    # Add ellipsis if truncated
    if len(sentences) > max_sentences:
        result += '...'

    return result.strip()
# -------------------------------------------------------------------
# 7. Thread Management
# -------------------------------------------------------------------
def get_thread_path(index_id: str, thread_id: str) -> Path:
    """Get path for thread JSON file"""
    return THREADS_DIR / index_id / f"{thread_id}.json"

def create_thread(index_id: str) -> str:
    """Create new thread and return thread ID"""
    thread_id = str(uuid.uuid4())
    thread_path = get_thread_path(index_id, thread_id)
    thread_path.parent.mkdir(parents=True, exist_ok=True)
    thread_path.touch()
    return thread_id

def load_thread_history(index_id: str, thread_id: str) -> List[Dict]:
    """Load and prune thread history"""
    thread_path = get_thread_path(index_id, thread_id)
    if not thread_path.exists():
        return []

    try:
        with open(thread_path, "r") as f:
            history = json.load(f)
    except:
        return []

    # Prune by time (1 hour)
    cutoff = time.time() - ONE_HOUR
    history = [msg for msg in history if msg['timestamp'] > cutoff]

    # Prune by message count (last 5 turns)
    history = history[-MAX_HISTORY_MESSAGES*2:]

    return history

def save_thread_message(index_id: str, thread_id: str, query: str, response: str):
    """Save new messages to thread history"""
    thread_path = get_thread_path(index_id, thread_id)
    history = load_thread_history(index_id, thread_id)
    
    history.append({
        "query": query,
        "response": response,
        "timestamp": time.time()
    })
    
    with open(thread_path, "w") as f:
        json.dump(history, f)
# -------------------------------------------------------------------
# 8. CSV Utility Functions
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
# 9. Embedding & Index Utility
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
    Returns a list of document chunks with metadata.
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
    """Create a FAISS index from document chunks."""
    texts = [doc["text"] for doc in documents]
    embeddings = embed_batch(texts)
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embeddings, index

def save_index_and_metadata(index_id: str, index: faiss.IndexFlatL2, documents: List[dict], embeddings: np.ndarray):
    """Save the FAISS index, embeddings, and document metadata."""
    index_path = INDEX_DIR / index_id
    index_path.mkdir(exist_ok=True)
    faiss.write_index(index, str(index_path / "index.faiss"))
    np.save(str(index_path / "embeddings.npy"), embeddings)
    with open(index_path / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)

def load_index_and_metadata(index_id: str) -> Tuple[faiss.IndexFlatL2, List[dict], np.ndarray]:
    """Load the FAISS index, embeddings, and document metadata."""
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
# 10. Index Lock Management
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
# 11. Initiate Chat
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
# 12. Background Task for Indexing
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
# 13. Async Endpoint: Upload File (Background Task)
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
    set_status_by_index_id(index_id, -1)  # Mark status = -1 (indexing in progress)

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
# 14. Conversation (Ephemeral) Storage
# -------------------------------------------------------------------
def load_conversation(index_id: str) -> List[dict]:
    """
    Load the existing conversation from threads/{index_id}.json.
    Returns a list of messages, each with { "role": "user"|"assistant", "content": "...", "timestamp": <float> }.
    """
    conv_file = THREADS_DIR / f"{index_id}.json"
    if not conv_file.exists():
        return []
    with open(conv_file, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return []

def save_conversation(index_id: str, messages: List[dict]):
    """
    Save the conversation to threads/{index_id}.json, overwriting old data.
    """
    conv_file = THREADS_DIR / f"{index_id}.json"
    conv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(conv_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def prune_old_messages(messages: List[dict], max_age=ONE_HOUR) -> List[dict]:
    """
    Keep only messages within the last `max_age` seconds.
    """
    now = time.time()
    return [m for m in messages if (now - m["timestamp"]) <= max_age]


# -------------------------------------------------------------------
# 15. Create thread for conversation history
# -------------------------------------------------------------------
@app.post("/create_thread", response_model=ThreadResponse)
def create_thread_endpoint(request: ThreadCreateRequest):
    """Create new conversation thread for an index"""
    if not request.index_id:
        raise HTTPException(400, "index_id is required")
    
    thread_id = create_thread(request.index_id)
    return ThreadResponse(
        thread_id=thread_id,
        created_at=time.time()
    )
# -------------------------------------------------------------------
# 16. Chat Endpoint (with ephemeral conversation & RAG)
# -------------------------------------------------------------------
@app.post("/chat/", response_model=ChatResponse)
async def chat_with_history(body: ChatRequestBody):
    try:
        response_text = ""
        sources = []
        start_time = time.time()
        
        # Mode 1: No index mode (normal chat)
        if not body.index_id:
            messages = [
                {"role": "system", "content": system_prompt + "\n\nProvide a well-structured response to the following query."},
                {"role": "user", "content": body.query}
            ]
            with llm_lock:
                outputs = llm.chat(messages, SamplingParams(temperature=0.7, max_tokens=1000))
            response_text = outputs[0].outputs[0].text.strip()
            return ChatResponse(
                response=response_text,
                stats={"time": round(time.time()-start_time, 2)}
            )

        # Mode 2/3: Index-based chat
        index, documents, _ = load_index_and_metadata(body.index_id)
        # Retrieve RAG context
        query_embed = embed_batch([body.query])[0].reshape(1, -1)
        faiss.normalize_L2(query_embed)
        _, idxs = index.search(query_embed, 5)
        sources = [documents[i]['source_file'] for i in idxs[0]]
        code_context = "\n".join(
            [f"File: {documents[i]['source_file']}\n{documents[i]['text']}" 
             for i in idxs[0]]
        )

        # Mode 2: Index without thread
        if not body.thread_id:
            usr_msg = f"""You have access to the following relevant code context:\n\n{code_context}\n\nUse this context to generate a clear and accurate response to the user's query."""
            messages = [
                {"role": "system", "content": system_prompt + '\n\n' + usr_msg},
                {"role": "user", "content": body.query}
            ]
        # Mode 3: Index with thread
        else:
            # Load and prepare conversation history
            history = load_thread_history(body.index_id, body.thread_id)
            conversation = []
            for msg in history:
                conversation.append(f"User: {msg['query']}")
                
                # Truncate response to three sentences
                # response_sentences = msg['response'].split(". ")
                # truncated_response = ". ".join(response_sentences[:3])  # Take first three sentences
                truncated_response = summarize_response(msg['response'])
                conversation.append(f"Assistant: {truncated_response}")
            conversation_history = '\n'.join(conversation[-MAX_HISTORY_MESSAGES*2:])
            user_msg = (
                "You have access to the following relevant code context:\n\n"
                f"{code_context}\n\n"
                "Additionally, here is the recent conversation history to maintain continuity:\n\n"
                f"{conversation_history}\n\n"
                "Use both the provided code context and past conversation to generate a precise and well-structured response to the user's query."
            )
            messages = [
                {"role": "system", "content": system_prompt + '\n\n' + user_msg},
                {"role": "user", "content": body.query}
            ]

        # Generate response
        with llm_lock:
            outputs = llm.chat(messages, SamplingParams(temperature=0.7, max_tokens=1000))
        response_text = outputs[0].outputs[0].text.strip()

        # Save to thread if applicable
        if body.thread_id:
            save_thread_message(
                body.index_id,
                body.thread_id,
                body.query,
                response_text
            )

        return ChatResponse(
            response=response_text,
            thread_id=body.thread_id,
            sources=sources,
            stats={"time": round(time.time()-start_time, 2)}
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
# -------------------------------------------------------------------
# 16. Main Entry
# -------------------------------------------------------------------
if __name__ == "__main__":
    # use --workers for concurrency: e.g. uvicorn main:app --workers 4
    uvicorn.run(app, host="0.0.0.0", port=8000)
