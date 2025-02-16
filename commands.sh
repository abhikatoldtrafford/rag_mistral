# Initiate chat 
curl -X POST "http://localhost:8000/initiate_chat/" \
-H "Content-Type: application/json" \
-d '{"user_id": "user123"}'

# Chat
curl -X POST "http://localhost:8000/chat/" \
-H "Content-Type: application/json" \
-d '{"query": "your question here", "index_id": "your_index_id"}'

# Upload file
curl -X POST "http://localhost:8000/upload_file_async/?index_id=your_index_id" \
-H "Content-Type: multipart/form-data" \
-F "zipfile=@/path/to/your/file.zip"

# Check upload status
curl -X POST "http://localhost:8000/upload_file_async/?index_id=your_index_id"