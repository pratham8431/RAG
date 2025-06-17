import asyncio
import json
import math
from flask import Blueprint, request, jsonify
from app.services.utils import (
    get_raw_data_pdf, get_raw_data_txt, get_raw_data_from_docx,
    get_raw_data_csv, get_raw_data_xlsx, recursive_chunker
)
from app.services.openai_services import (
    store_vector_data_azure, process_user_query,
)
from app.services.gemini_services import generate_influencer_list
import os
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use absolute path for embeddings directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, '..', '..', 'embeddings')
STORAGE_DIR = os.path.join(BASE_DIR, '..', '..', 'storage')

rag_routes = Blueprint('rag_routes', __name__)

@rag_routes.route('/upload', methods=['POST'])
async def upload_document():
    try:
        start_time = time.time()
        file = request.files['file']
        uuid_val = str(uuid.uuid4())
        file_name = file.filename

        # Validate file extension
        if not any(file_name.lower().endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.csv', '.xlsx', '.xls']):
            logger.error("Unsupported file extension")
            return jsonify({"success": False, "message": "Unsupported file extension"}), 400

        # Save file locally
        storage_dir = os.path.join(STORAGE_DIR, uuid_val)
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, file_name)
        file.save(file_path)
        logger.info(f"File saved: {file_path}, Time: {time.time() - start_time:.2f}s")

        # Extract text
        extract_start = time.time()
        if file_name.lower().endswith('.txt'):
            raw_text = get_raw_data_txt(file_path)
        elif file_name.lower().endswith('.pdf'):
            raw_text = get_raw_data_pdf(file_path)
        elif file_name.lower().endswith('.docx'):
            raw_text = get_raw_data_from_docx(file_path)
        elif file_name.lower().endswith('.csv'):
            raw_text = get_raw_data_csv(file_path)
        elif file_name.lower().endswith(('.xlsx', '.xls')):
            raw_text = get_raw_data_xlsx(file_path)
        else:
            logger.error("File extension validation failed after initial check")
            return jsonify({"success": False, "message": "File extension validation failed"}), 400
        logger.info(f"Text extracted, Time: {time.time() - extract_start:.2f}s")
        logger.info(f"Extracted text (first 100 chars): {str(raw_text)[:100] if raw_text else 'None'}")

        # Chunk text
        chunk_start = time.time()
        text_chunks = recursive_chunker(raw_text)
        logger.info(f"Text chunked ({len(text_chunks)} chunks), Time: {time.time() - chunk_start:.2f}s")

        # Store embeddings
        embedding_dir = os.path.join(EMBEDDINGS_DIR, uuid_val)
        os.makedirs(embedding_dir, exist_ok=True)
        embed_start = time.time()
        await store_vector_data_azure(text_chunks, embedding_dir)
        logger.info(f"Embeddings stored, Time: {time.time() - embed_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Total upload time: {total_time:.2f}s")

        return jsonify({
            "success": True,
            "message": "File uploaded and processed successfully",
            "uuid": uuid_val
        }), 200

    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@rag_routes.route('/qna', methods=['POST'])
async def qna():
    try:
        data = request.get_json()
        uuid_val = data.get('uuid')  # Optional
        question = data.get('question')

        if not question:
            return jsonify({"success": False, "message": "Question is required"}), 400

        # Initialize response structure
        response_data = {
            "success": True,
            "document_used": False,
            "search_parameters": None,
            "logs": []
        }

        # Process the query
        processing_uuid = uuid_val if uuid_val else "default_"+str(uuid.uuid4())
        ai_response = await process_user_query(processing_uuid, question, [])
        
        # Handle the response structure
        if 'error' in ai_response:
            response_data["success"] = False
            response_data["error"] = ai_response.get("error", "Unknown error")
        
        # Update response data from AI response
        response_data["document_used"] = ai_response.get("context_used", False)
        response_data["logs"] = ai_response.get("logs", [])
        
        # Handle search parameters
        if 'search_parameters' in ai_response:
            if isinstance(ai_response['search_parameters'], dict):
                response_data["search_parameters"] = ai_response['search_parameters']
            else:
                try:
                    response_data["search_parameters"] = json.loads(ai_response['search_parameters'])
                except (json.JSONDecodeError, TypeError):
                    response_data["search_parameters"] = {
                        "raw_response": str(ai_response['search_parameters']),
                        "note": "Response formatting issue"
                    }
        else:
            response_data["search_parameters"] = None
            response_data["success"] = False
            response_data["error"] = "No search parameters generated"

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in QnA: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Internal server error",
            "error": str(e),
            "logs": [{"type": "error", "message": f"System error: {str(e)}"}]
        }), 500
    
@rag_routes.route('/discover-influencers', methods=['POST'])
async def discover_influencers():
    """
    Endpoint to discover influencers based on search parameters
    """
    try:
        data = request.get_json()
        search_params = data.get('search_parameters')
        
        if not search_params:
            return jsonify({
                "success": False,
                "message": "Search parameters are required"
            }), 400

        # Generate influencer list using Gemini
        result = await generate_influencer_list(search_params)
        
        if not result.get('success'):
            return jsonify({
                "success": False,
                "message": result.get('message', 'Failed to generate influencers'),
                "error": result.get('error')
            }), 500

        return jsonify({
            "success": True,
            "count": result.get('count', 0),
            "influencers": result.get('influencers', []),
            "logs": result.get('logs', [])
        })

    except Exception as e:
        logger.error(f"Error in influencer discovery: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Internal server error",
            "error": str(e)
        }), 500