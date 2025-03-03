import os
import uuid
import re
import glob
import time
from typing import Optional, List
from langchain_openai import ChatOpenAI

# PDF extraction
from pdfminer.high_level import extract_text

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# For pretty printing
from rich import print

# Pinecone client using the new API:
from pinecone import Pinecone, ServerlessSpec

# Hugging Face for embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# LangChain imports for LLM summarization
from langchain.prompts import ChatPromptTemplate
# Note: ChatOpenAI is deprecated; consider updating if possible.
from langchain_community.chat_models.openai import ChatOpenAI
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic

# Import OpenAI error handling
try:
    from openai.error import RateLimitError
except ModuleNotFoundError:
    print("[yellow]Warning: Module 'openai.error' not found. Please install it via pip.[/yellow]")
    class RateLimitError(Exception):
        pass

# -----------------------------------------------------------------------------
# Helper: Truncate Text
# -----------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate the text to a maximum number of characters."""
    return text if len(text) <= max_chars else text[:max_chars] + "..."

# -----------------------------------------------------------------------------
# Helper: Safe Invoke with Retry Mechanism
# -----------------------------------------------------------------------------

def safe_invoke(func, params, max_retries=3, delay=5):
    """Call a function with retries if a RateLimitError occurs."""
    for attempt in range(max_retries):
        try:
            return func(params)
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise Exception("Insufficient quota. Check your OpenAI billing plan. " + str(e))
            print(f"[red]Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt+1}/{max_retries})[/red]")
            time.sleep(delay)
    raise Exception("Maximum retries exceeded due to rate limits.")

# -----------------------------------------------------------------------------
# Helper: Batch Upsert Vectors
# -----------------------------------------------------------------------------

def batch_upsert_vectors(index, vectors: List[dict], batch_size: int = 10):
    """Upsert vectors in batches to keep payload sizes below Pinecone’s limit."""
    responses = []
    total = len(vectors)
    for i in range(0, total, batch_size):
        batch = vectors[i:i+batch_size]
        response = index.upsert(vectors=batch)
        responses.append(response)
        print(f"[bold green]Upserted batch {i//batch_size + 1} of {((total - 1)//batch_size) + 1}[/bold green]")
    return responses

# =============================================================================
# Functions for PDF extraction & text cleaning
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"[red]Error extracting text from {pdf_path}: {e}[/red]")
        return ""

def clean_text(text: str) -> str:
    """Removes unwanted characters and normalizes whitespace."""
    text = re.sub(r'[^A-Za-z0-9\s.,;:()\-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# =============================================================================
# Agentic Chunker Class
# =============================================================================
# This class splits the entire text into chunks using a two-step approach:
# 1. It first splits the text by page breaks ("\x0c"). If no page breaks exist,
#    it falls back to splitting on double newlines.
# 2. Then it uses a sliding-window method to create overlapping chunks that cover the full text.
# Finally, each chunk is enriched with a summary and title via the LLM.
class AgenticChunker:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.id_truncate_limit = 5  # Short chunk IDs for simplicity.
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY not provided in environment variables.")
        self.llm = ChatOpenAI(model_name='gpt-4-turbo', openai_api_key=openai_api_key, temperature=0.2)

    def chunk_document(self, text: str, target_chars: int = 2000, overlap_chars: int = 200) -> List[dict]:
        """Splits the text into overlapping chunks that cover the full document."""
        # First, try splitting by page breaks (form feed character)
        if "\x0c" in text:
            segments = text.split("\x0c")
        else:
            segments = text.split("\n\n")
        full_text = " ".join(segments)
        chunks = []
        start = 0
        text_length = len(full_text)
        while start < text_length:
            end = start + target_chars
            chunk = full_text[start:end]
            chunks.append(chunk)
            start = max(0, start + target_chars - overlap_chars)
        # Enrich each chunk with summary and title
        chunk_dicts = []
        for chunk_text in chunks:
            summary = self._get_new_chunk_summary(chunk_text)
            title = self._get_new_chunk_title(summary)
            chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
            chunk_dicts.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'summary': summary,
                'title': title
            })
        return chunk_dicts

    def _get_new_chunk_summary(self, text: str) -> str:
        truncated_text = truncate_text(text, 1000)
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Generate a concise 1-sentence summary capturing the main ideas of the following text chunk."),
            ("user", "Text chunk:\n{text}")
        ])
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {"text": truncated_text})
        return result.content

    def _get_new_chunk_title(self, summary: str) -> str:
        truncated_summary = truncate_text(summary, 500)
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Based on the following summary, generate a brief title that captures the main topic."),
            ("user", "Summary:\n{summary}")
        ])
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {"summary": truncated_summary})
        return result.content

# =============================================================================
# Embedding Function
# =============================================================================
# We use the Hugging Face model "dwzhu/e5-base-4k" which produces 768-dimensional embeddings.
EMBEDDING_MODEL_NAME = "dwzhu/e5-base-4k"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

def get_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()

# =============================================================================
# Initialize Pinecone Index
# =============================================================================
# IMPORTANT: The index dimension is set to 768 to match the embedding output.
def init_pinecone_index(index_name: str, dimension: int = 768):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not set in environment variables")
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        # Check the dimension of the existing index.
        desc = pc.describe_index(index_name)
        existing_dim = desc["dimension"]
        if existing_dim != dimension:
            print(f"Index '{index_name}' exists with dimension {existing_dim}.")
            print("Deletion protection is enabled. Please disable deletion protection via the Pinecone dashboard and delete the index, or use a new index name.")
            raise ValueError(f"Index '{index_name}' dimension mismatch: expected {dimension}, found {existing_dim}.")
    else:
        print(f"Index '{index_name}' does not exist. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2'),
            deletion_protection=False
        )
    index = pc.Index(index_name)
    return index

# =============================================================================
# Main: Process PDFs and Upsert Chunks to Pinecone
# =============================================================================

def main():
    pdf_folder_path = "./books"  # PDFs should be placed in a folder named "books"
    if not os.path.isdir(pdf_folder_path):
        print(f"[red]PDF folder not found: {pdf_folder_path}[/red]")
        return

    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not pdf_files:
        print(f"[red]No PDF files found in folder: {pdf_folder_path}[/red]")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"[bold green]Processing PDF:[/bold green] {pdf_file}")
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text:
            continue
        cleaned_text = clean_text(raw_text)
        agentic_chunker = AgenticChunker()
        pdf_chunks = agentic_chunker.chunk_document(cleaned_text, target_chars=2000, overlap_chars=200)
        print(f"[bold blue]Created {len(pdf_chunks)} chunks from {os.path.basename(pdf_file)}.[/bold blue]")
        all_chunks.extend(pdf_chunks)

    if not all_chunks:
        print("[red]No chunks extracted from any PDFs.[/red]")
        return

    # Initialize Pinecone index (using dimension=768).
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
    pinecone_index = init_pinecone_index(index_name, dimension=768)

    vectors = []
    for chunk in all_chunks:
        embedding = get_embedding(chunk['text'])
        truncated_text = truncate_text(chunk['text'], 200)
        vector = {
            "id": chunk['chunk_id'],
            "values": embedding,
            "metadata": {
                "title": chunk['title'],
                "summary": chunk['summary'],
                "text": truncated_text
            }
        }
        vectors.append(vector)

    batch_upsert_vectors(pinecone_index, vectors, batch_size=10)
    print(f"[bold green]Finished upserting vectors into Pinecone index '{index_name}'.[/bold green]")

if __name__ == "__main__":
    main()