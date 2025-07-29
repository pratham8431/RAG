# RAG (Retrieval-Augmented Generation) API

A Flask-based backend for document ingestion, semantic search, and influencer discovery using OpenAI and Google Gemini APIs. This project enables users to upload documents, generate embeddings, perform question answering over their data, and discover influencers for marketing campaigns.

## Features

- **Document Upload & Embedding:** Upload `.txt`, `.pdf`, `.docx`, `.csv`, or `.xlsx` files. The system extracts, chunks, and embeds the content using OpenAI’s embedding models and stores them with FAISS for efficient retrieval.
- **Semantic Q&A:** Ask questions about your uploaded documents. The system retrieves relevant chunks and generates answers using OpenAI’s chat models.
- **Influencer Discovery:** Generate a list of influencers matching campaign needs using Google Gemini’s generative AI.
- **Async Flask API:** All endpoints are asynchronous for high performance.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   - `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, `AZURE_DEPLOYMENT_EMBEDDING` for OpenAI.
   - `GEMINI_API_KEY` for Google Gemini.
   - Optionally, `SECRET_KEY` and `FLASK_ENV`.

4. **Run the server:**
   ```bash
   python run.py
   ```
   The API will be available at `http://localhost:5000`.

## API Endpoints

### 1. Upload Document

**POST** `/upload`

- Upload a document for embedding and storage.
- **Body (form-data):** `file` (the document file)
- **Response:** `{ "success": true, "message": "...", "uuid": "<doc_id>" }`

### 2. Question Answering

**POST** `/qna`

- Ask a question about an uploaded document.
- **Body (JSON):**
  ```json
  {
    "uuid": "<doc_id>",   // optional, if omitted uses default
    "question": "Your question here"
  }
  ```
- **Response:** Answer, context usage, logs, and search parameters.

### 3. Discover Influencers

**POST** `/discover-influencers`

- Generate a list of influencers based on campaign/search parameters.
- **Body (JSON):**
  ```json
  {
    "search_parameters": { /* campaign criteria */ }
  }
  ```
- **Response:** List of influencers, count, and logs.

## File Structure

- `app/` - Main application code (routes, services)
- `embeddings/` - Stores FAISS indices and chunk data
- `storage/` - Stores uploaded files
- `config.py` - Configuration
- `run.py` - Entrypoint

## License

MIT (or specify your license) 