# Initiate chat 
curl -X POST "http://localhost:8000/initiate_chat/" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123"}'

# Chat
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this repository all about?", "index_id": "692900d6-e6b2-4c99-b936-25456d89e4ab"}'


# Upload file
curl -X POST "http://localhost:8000/upload_file_async/?index_id=692900d6-e6b2-4c99-b936-25456d89e4ab" \
-H "Content-Type: multipart/form-data" \
-F "zipfile=@ultralytics-main.zip"

# Check upload status
curl -X POST "http://localhost:8000/upload_file_async/?index_id=692900d6-e6b2-4c99-b936-25456d89e4ab"

#create thread
curl -X POST "http://localhost:8000/create_thread" \
     -H "Content-Type: application/json" \
     -d '{
           "index_id": "692900d6-e6b2-4c99-b936-25456d89e4ab"
         }'

# Chat with thread
curl -X POST "http://localhost:8000/chat/" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "How does this repository work?",
           "index_id": "692900d6-e6b2-4c99-b936-25456d89e4ab",
           "thread_id": "3ae85763-4f0f-4ffb-aad7-99d69da021a3"
         }'
