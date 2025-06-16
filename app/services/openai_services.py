import openai
import os
import json
import numpy as np
from faiss import IndexFlatL2
import pickle
from typing import List, Dict, Any
import asyncio
import logging
import time
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def store_vector_data_azure(text_chunks: List[str], embedding_dir: str) -> None:
    start_time = time.time()
    client = None
    try:
        client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            timeout=httpx.Timeout(30.0),
            http_client=httpx.AsyncClient(proxies=None)
        )

        # Batch embeddings to reduce API calls
        batch_size = 10
        embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            batch_start = time.time()
            try:
                response = await client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"  # Using the correct embedding model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"Batch {i//batch_size + 1} embeddings generated, Time: {time.time() - batch_start:.2f}s")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                raise

        # Save embeddings to FAISS index
        faiss_start = time.time()
        embeddings = np.array(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]
        index = IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"FAISS index created, Time: {time.time() - faiss_start:.2f}s")

        # Save index and chunks
        os.makedirs(embedding_dir, exist_ok=True)
        with open(os.path.join(embedding_dir, 'index.faiss'), 'wb') as f:
            pickle.dump(index, f)
        with open(os.path.join(embedding_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(text_chunks, f)

        logger.info(f"Total embedding storage time: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"Failed to store vector data: {str(e)}")
        raise
    finally:
        if client:
            await client.close()

async def process_user_query(uuid: str, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, str]:
    if chat_history is None:
        chat_history = []
        
    embedding_client = None
    chat_client = None
    logs = []
    
    try:
        logs.append({"type": "info", "message": "Initializing AI services..."})
        
        # Initialize clients
        embedding_client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            timeout=httpx.Timeout(30.0),
            http_client=httpx.AsyncClient(proxies=None)
        )
        
        chat_client = openai.AsyncAzureOpenAI(
            api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION'),
            azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
            timeout=httpx.Timeout(30.0),
            http_client=httpx.AsyncClient(proxies=None)
        )

        context = ""
        if uuid and uuid != "default":
            embedding_dir = os.path.join("embeddings", uuid)
            if os.path.exists(embedding_dir):
                logs.append({"type": "info", "message": "Analyzing documents..."})
                
                with open(os.path.join(embedding_dir, 'index.faiss'), 'rb') as f:
                    index = pickle.load(f)
                with open(os.path.join(embedding_dir, 'chunks.pkl'), 'rb') as f:
                    text_chunks = pickle.load(f)

                logs.append({"type": "info", "message": "Processing query..."})
                query_response = await embedding_client.embeddings.create(
                    input=query,
                    model="text-embedding-3-small"
                )
                query_embedding = np.array([query_response.data[0].embedding], dtype=np.float32)

                logs.append({"type": "info", "message": "Searching documents..."})
                distances, indices = index.search(query_embedding, k=3)
                context = "\n".join([text_chunks[idx] for idx in indices[0]])
                logs.append({"type": "info", "message": f"Found {len(indices[0])} relevant sections"})
            else:
                logs.append({"type": "warning", "message": "No documents found"})
        else:
            logs.append({"type": "info", "message": "No documents provided"})

        # Strict JSON prompt
        prompt = f"""Convert this request into structured JSON for influencer search:

REQUEST: {query}
{f"CONTEXT: {context}" if context else ""}

OUTPUT ONLY THIS JSON STRUCTURE (NO OTHER TEXT):
{{
  "campaign": {{
    "objective": "awareness|engagement|conversion",
    "duration": "1-2 weeks|1 month|2+ months"
  }},
  "product": {{
    "category": "beauty|fashion|tech|etc",
    "key_features": ["feature1", "feature2"]
  }},
  "audience": {{
    "age_range": [min, max],
    "gender": "male|female|unisex",
    "location": "country/region",
    "interests": ["interest1", "interest2"]
  }},
  "platforms": ["Instagram", "TikTok", "YouTube"],
  "content_types": ["posts", "stories", "videos"],
  "influencer": {{
    "type": "nano|micro|macro|celebrity",
    "count": number,
    "metrics": {{
      "followers": "min-max",
      "engagement": "min%"
    }}
  }},
  "keywords": {{
    "primary": ["keyword1", "keyword2"],
    "secondary": ["keyword3", "keyword4"]
  }}
}}"""

        logs.append({"type": "info", "message": "Generating structured response..."})
        response = await chat_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON generator. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Clean and parse response
        raw_response = response.choices[0].message.content
        try:
            search_params = json.loads(raw_response)
            logs.append({"type": "success", "message": "Response generated"})
            return {
                "search_parameters": search_params,
                "logs": logs,
                "context_used": bool(context)
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logs.append({"type": "warning", "message": "Formatting response"})
            return {
                "search_parameters": {
                    "raw_response": raw_response,
                    "note": "Response formatting failed"
                },
                "logs": logs,
                "context_used": bool(context)
            }
    
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg)
        logs.append({"type": "error", "message": error_msg})
        return {
            "search_parameters": {
                "error": "Processing failed",
                "details": str(e)
            },
            "logs": logs,
            "error": True
        }
    finally:
        if embedding_client:
            await embedding_client.close()
        if chat_client:
            await chat_client.close()