import streamlit as st
import requests
import time

# -------------------------------------------
# Set Page Config (must be first)
# -------------------------------------------
st.set_page_config(page_title="RAG Chat", layout="wide")

# -------------------------------------------
# Configuration
# -------------------------------------------
API_BASE_URL = "http://localhost:8000"  # Update if needed

# -------------------------------------------
# Session State Initialization
# -------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "index_id" not in st.session_state:
    st.session_state.index_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_zip" not in st.session_state:
    st.session_state.uploaded_zip = None
if "file_status" not in st.session_state:
    st.session_state.file_status = None

# Create persistent placeholders if not already created
if "file_status_placeholder" not in st.session_state:
    st.session_state.file_status_placeholder = st.empty()
if "chat_placeholder" not in st.session_state:
    st.session_state.chat_placeholder = st.empty()

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def initiate_session(user_id: str):
    """Call /initiate_chat/ to start a session (or retrieve an existing one)."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/initiate_chat/",
            json={"user_id": user_id}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.user_id = user_id
            st.session_state.index_id = data["index_id"]
            st.success(f"Welcome back, {user_id}! Attached index: {st.session_state.index_id}")
        else:
            st.error(f"Session start failed: {response.text}")
    except Exception as e:
        st.error(f"Error initiating session: {str(e)}")

def upload_zip(zip_file):
    """Upload a ZIP file asynchronously and update status in real-time."""
    if zip_file is None or not st.session_state.index_id:
        st.error("Please start a session first and select a file.")
        return

    try:
        st.session_state.file_status_placeholder.info("Uploading... Indexing will start in the background.")
        files = {"zipfile": (zip_file.name, zip_file.getvalue(), "application/zip")}
        response = requests.post(f"{API_BASE_URL}/upload_file_async/?index_id={st.session_state.index_id}", files=files)
        
        if response.status_code == 200:
            st.session_state.uploaded_zip = zip_file.name
            st.session_state.file_status = "Indexing started in background."
            st.session_state.file_status_placeholder.info("Indexing started in background. Check status below.")
        else:
            st.session_state.file_status_placeholder.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.session_state.file_status_placeholder.error(f"Error uploading file: {str(e)}")

def check_index_status():
    """Check the current indexing status and persist the message in UI."""
    if not st.session_state.index_id:
        st.error("No active session.")
        return

    with st.spinner("Checking index status..."):
        try:
            response = requests.post(f"{API_BASE_URL}/upload_file_async/?index_id={st.session_state.index_id}")
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")

                if status == -1:
                    st.session_state.file_status = "⏳ Indexing is still in progress. Please wait..."
                    st.session_state.file_status_placeholder.warning(st.session_state.file_status)
                elif status == 0:
                    st.session_state.file_status = "✅ Indexing is complete! You can start chatting."
                    st.session_state.file_status_placeholder.success(st.session_state.file_status)
                else:
                    st.session_state.file_status = f"ℹ️ Status: {status}"
                    st.session_state.file_status_placeholder.info(st.session_state.file_status)

            else:
                st.session_state.file_status = f"❌ Status check failed: {response.text}"
                st.session_state.file_status_placeholder.error(st.session_state.file_status)

        except Exception as e:
            st.session_state.file_status = f"❌ Error checking status: {str(e)}"
            st.session_state.file_status_placeholder.error(st.session_state.file_status)


def send_message(query: str):
    """Send chat message to /chat/ endpoint and show 'typing' indicator."""
    if not st.session_state.index_id:
        st.error("Please start a session and upload a ZIP file first.")
        return

    # Append user message to history immediately
    st.session_state.chat_history.append({"role": "user", "content": query})
    # Update chat display so the user sees their message
    update_chat_display()

    # Create a local placeholder for the assistant's response
    assistant_placeholder = st.empty()
    assistant_placeholder.text("Typing...")

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json={
                "query": query,
                "index_id": st.session_state.index_id
            }
        )
        if response.status_code == 200:
            data = response.json()
            assistant_response = data.get("response", "")
            # Clear the temporary placeholder
            assistant_placeholder.empty()
            # Append the assistant's response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_response,
                "sources": data.get("context_sources", [])
            })
            update_chat_display()
        else:
            assistant_placeholder.error(f"Error in API response: {response.text}")
    except Exception as e:
        assistant_placeholder.error(f"Communication error: {str(e)}")

def update_chat_display():
    """Update the chat area using the persistent chat placeholder."""
    chat_container = st.session_state.chat_placeholder
    chat_container.empty()  # Clear previous content
    with chat_container.container():
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show sources only if they exist and if the content is not a generic fallback
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("🔎 View Sources", expanded=False):
                        for source in message["sources"]:
                            st.code(source, language="plaintext")

def reset_session():
    """Reset session and clear history."""
    try:
        if st.session_state.index_id:
            requests.post(f"{API_BASE_URL}/clear_chat/", json={"index_id": st.session_state.index_id})
    except Exception:
        pass
    st.session_state.user_id = None
    st.session_state.index_id = None
    st.session_state.chat_history = []
    st.session_state.uploaded_zip = None
    st.session_state.file_status = None
    st.session_state.file_status_placeholder.empty()
    st.session_state.chat_placeholder.empty()
    st.experimental_rerun()

# -------------------------------------------
# Sidebar UI
# -------------------------------------------
with st.sidebar:
    st.header("Session Setup")
    if not st.session_state.user_id:
        user_id_input = st.text_input("Enter your user ID:")
        if st.button("Start Session"):
            if user_id_input:
                initiate_session(user_id_input)
            else:
                st.error("Please enter a user ID.")
    else:
        st.subheader(f"Welcome back, {st.session_state.user_id}!")
        st.write(f"Attached index: `{st.session_state.index_id}`")

    st.divider()
    st.header("Repository Setup")
    if st.session_state.index_id:
        uploaded_file = st.file_uploader(
            "Upload ZIP Repository",
            type=["zip"],
            accept_multiple_files=False,
            key="zip_uploader"
        )
        if uploaded_file and uploaded_file.name != st.session_state.uploaded_zip:
            upload_zip(uploaded_file)
        if st.session_state.uploaded_zip:
            st.write(f"📁 {st.session_state.uploaded_zip}")
        if st.button("Check Index Status"):
            check_index_status()

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()

# -------------------------------------------
# Main Chat Interface
# -------------------------------------------
st.title("💬 Repository Chat")
st.caption("Chat with your code repository using RAG")

# Chat input at the bottom
chat_input = st.chat_input("Ask about your repository...")
if chat_input:
    send_message(chat_input)

# Update chat display if there are existing messages
if st.session_state.chat_history:
    update_chat_display()
