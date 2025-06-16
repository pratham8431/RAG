import pypdf
from docx2txt import process as docx_process
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_raw_data_pdf(file_path: str) -> str:
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

def get_raw_data_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_raw_data_from_docx(file_path: str) -> str:
    return docx_process(file_path)

def get_raw_data_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_string()

def get_raw_data_xlsx(file_path: str) -> str:
    df = pd.read_excel(file_path)
    return df.to_string()

def recursive_chunker(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    logger.info("Starting text chunking")
    if not isinstance(text, str):
        logger.error(f"Invalid input for chunking: text is of type {type(text)}, expected str")
        return []
    if not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    chunks = []
    text = text.strip()
    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        # Find the nearest space to avoid splitting words
        if end < text_length:
            while end > start and text[end] != " ":
                end -= 1
            if end == start:  # No space found, force split at chunk_size
                end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        logger.debug(f"Chunk created: {chunk[:50]}... (length: {len(chunk)})")
        start = end - overlap if end < text_length else end

    logger.info(f"Chunking completed, {len(chunks)} chunks created")
    return chunks