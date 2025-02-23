import streamlit as st
import requests
import time
import uuid
from datetime import datetime

# -------------------------------------------
# Set Page Config
# -------------------------------------------
st.set_page_config(page_title="RAG Chat", layout="wide")

# -------------------------------------------
# Configuration
# -------------------------------------------
API_BASE_URL = "http://localhost:8000"

# -------------------------------------------
# Session State Initialization
# -------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "index_id" not in st.session_state:
    st.session_state.index_id = None
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread" not in st.session_state:
    st.session_state.current_thread = None
if "uploaded_zip" not in st.session_state:
    st.session_state.uploaded_zip = None
if "file_status" not in st.session_state:
    st.session_state.file_status = None

# Persistent placeholders
if "file_status_placeholder" not in st.session_state:
    st.session_state.file_status_placeholder = st.empty()
if "chat_placeholder" not in st.session_state:
    st.session_state.chat_placeholder = st.empty()

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def initiate_session(user_id: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/initiate_chat/",
            json={"user_id": user_id}
        )
        if response.ok:
            data = response.json()
            st.session_state.user_id = user_id
            st.session_state.index_id = data["index_id"]
            st.success(f"Session started! Index ID: {st.session_state.index_id}")
            # Create default thread
            create_new_thread("General")
        else:
            st.error(f"Session start failed: {response.text}")
    except Exception as e:
        st.error(f"Error initiating session: {str(e)}")

def create_new_thread(thread_name: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/create_thread",
            json={"index_id": st.session_state.index_id}
        )
        if response.ok:
            thread_id = response.json()["thread_id"]
            st.session_state.threads[thread_id] = {
                "name": thread_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": []
            }
            st.session_state.current_thread = thread_id
            st.success(f"New thread created: {thread_name}")
        else:
            st.error(f"Thread creation failed: {response.text}")
    except Exception as e:
        st.error(f"Error creating thread: {str(e)}")

def upload_zip(zip_file):
    """Upload ZIP only if it hasn't been uploaded before."""
    if zip_file is None or not st.session_state.index_id:
        st.error("Please start a session first and select a file.")
        return

    # ✅ NEW CHECK: Prevent re-uploading the same file
    if st.session_state.uploaded_zip == zip_file.name:
        st.session_state.file_status_placeholder.info("✅ File already uploaded. No need to re-upload.")
        return

    try:
        st.session_state.file_status_placeholder.info("Uploading... Indexing will start in the background.")
        files = {"zipfile": (zip_file.name, zip_file.getvalue(), "application/zip")}
        response = requests.post(f"{API_BASE_URL}/upload_file_async/?index_id={st.session_state.index_id}", files=files)

        if response.ok:
            st.session_state.uploaded_zip = zip_file.name  # ✅ Store the filename to avoid duplicate uploads
            st.session_state.file_status = "Indexing started in background."
            st.session_state.file_status_placeholder.info("Indexing started. Check status below.")
        else:
            st.session_state.file_status_placeholder.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.session_state.file_status_placeholder.error(f"Error uploading file: {str(e)}")


def check_index_status():
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload_file_async/?index_id={st.session_state.index_id}"
        )
        if response.ok:
            status = response.json().get("status")
            if status == -1:
                st.session_state.file_status_placeholder.warning("⏳ Indexing in progress...")
            elif status == 0:
                st.session_state.file_status_placeholder.success("✅ Indexing complete!")
        else:
            st.session_state.file_status_placeholder.error(f"Status check failed: {response.text}")
    except Exception as e:
        st.session_state.file_status_placeholder.error(f"Error checking status: {str(e)}")

def send_message(query: str):
    try:
        # Prepare request data
        data = {"query": query}
        if st.session_state.index_id:
            data["index_id"] = st.session_state.index_id
        if st.session_state.current_thread:
            data["thread_id"] = st.session_state.current_thread

        # Add temporary message
        temp_id = str(uuid.uuid4())
        st.session_state.threads[st.session_state.current_thread]["messages"].append({
            "role": "user",
            "content": query,
            "temp_id": temp_id,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        update_chat_display()

        # Send request
        response = requests.post(f"{API_BASE_URL}/chat/", json=data)
        if response.ok:
            response_data = response.json()
            # Update message with final response
            for idx, msg in enumerate(st.session_state.threads[st.session_state.current_thread]["messages"]):
                if msg.get("temp_id") == temp_id:
                    st.session_state.threads[st.session_state.current_thread]["messages"][idx] = {
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().strftime("%H:%M")
                    }
                    st.session_state.threads[st.session_state.current_thread]["messages"].append({
                        "role": "assistant",
                        "content": response_data["response"],
                        "sources": response_data.get("sources", []),
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    break
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Communication error: {str(e)}")
    finally:
        update_chat_display()

def update_chat_display():
    chat_container = st.session_state.chat_placeholder
    chat_container.empty()
    
    with chat_container.container():
        if st.session_state.current_thread:
            messages = st.session_state.threads[st.session_state.current_thread]["messages"]
            for msg in messages:
                with st.chat_message(msg["role"]):
                    col1, col2 = st.columns([10, 2])
                    with col1:
                        st.markdown(msg["content"])
                    with col2:
                        st.caption(msg["timestamp"])
                    
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("🔍 Sources"):
                            for source in msg["sources"]:
                                st.code(source, language="plaintext")

def reset_session():
    st.session_state.user_id = None
    st.session_state.index_id = None
    st.session_state.threads = {}
    st.session_state.current_thread = None
    st.session_state.uploaded_zip = None
    st.session_state.file_status = None
    st.session_state.file_status_placeholder.empty()
    st.session_state.chat_placeholder.empty()

# -------------------------------------------
# UI Components
# -------------------------------------------
# Sidebar
with st.sidebar:
    st.header("Session Management")
    
    if not st.session_state.user_id:
        user_id = st.text_input("Enter User ID:")
        if st.button("Start Session"):
            if user_id:
                initiate_session(user_id)
            else:
                st.error("Please enter a user ID")
    else:
        st.subheader(f"User: {st.session_state.user_id}")
        st.write(f"Index ID: `{st.session_state.index_id}`")
        
        # Thread management
        st.divider()
        st.subheader("Threads")
        
        # New thread creation
        new_thread_name = st.text_input("New Thread Name", placeholder="Enter thread name")
        if st.button("➕ Create New Thread"):
            if new_thread_name:
                create_new_thread(new_thread_name)
            else:
                st.error("Please enter a thread name")
        
        # Thread selection
        if st.session_state.threads:
            thread_options = {
                f"{details['name']} ({details['created_at']})": thread_id
                for thread_id, details in st.session_state.threads.items()
            }
            selected_thread = st.selectbox(
                "Select Thread",
                options=list(thread_options.keys()),
                index=0
            )
            st.session_state.current_thread = thread_options[selected_thread]

    # File upload
    st.divider()
    st.subheader("Repository Upload")
    if st.session_state.index_id:
        zip_file = st.file_uploader("Upload ZIP", type=["zip"])
        if zip_file:
            upload_zip(zip_file)
        if st.button("Check Index Status"):
            check_index_status()

    st.divider()
    if st.button("Reset Session", type="primary"):
        reset_session()

# Main interface
st.title("💬 Repository Chat Assistant")
st.caption("Chat with your code repository using RAG with thread support")

# Chat input
if prompt := st.chat_input("Ask about your repository..."):
    send_message(prompt)

# Display chat history
update_chat_display()