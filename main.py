import os
import uuid
import re
import glob
import time
from typing import Optional, List

# PDF extraction
from pdfminer.high_level import extract_text

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# For pretty printing
from rich import print

# Pinecone for vector database using the new API:
from pinecone import Pinecone, ServerlessSpec

# Hugging Face for embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# LangChain imports
from langchain.prompts import ChatPromptTemplate
# Use updated import from langchain_community as per deprecation notice
from langchain_community.chat_models.openai import ChatOpenAI
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic

# Import OpenAI error handling with fallback if not installed.
try:
    from openai.error import RateLimitError
except ModuleNotFoundError:
    print(
        "[yellow]Warning: Module 'openai.error' not found. Please install 'openai' via pip for proper rate limit handling.[/yellow]")


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
    """
    Calls the given function with params.
    If a RateLimitError occurs, checks for 'insufficient_quota' in the error message
    and immediately raises an exception; otherwise, retries the call.
    """
    for attempt in range(max_retries):
        try:
            result = func(params)
            return result
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise Exception("Insufficient quota. Check your OpenAI billing plan. " + str(e))
            print(
                f"[red]Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})[/red]")
            time.sleep(delay)
    raise Exception("Maximum retries exceeded due to rate limits.")


# -----------------------------------------------------------------------------
# Helper: Batch Upsert Vectors
# -----------------------------------------------------------------------------

def batch_upsert_vectors(index, vectors: List[dict], batch_size: int = 10):
    """Upserts vectors in batches to avoid exceeding request size limits."""
    responses = []
    total = len(vectors)
    for i in range(0, total, batch_size):
        batch = vectors[i:i + batch_size]
        response = index.upsert(vectors=batch)
        responses.append(response)
        print(f"[bold green]Upserted batch {i // batch_size + 1} of {((total - 1) // batch_size) + 1}[/bold green]")
    return responses


# =============================================================================
# 1. FUNCTIONS FOR PDF EXTRACTION & TEXT CLEANING
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"[red]Error extracting text from {pdf_path}: {e}[/red]")
        return ""


def clean_text(text: str) -> str:
    """Cleans text by removing unwanted characters and normalizing whitespace."""
    text = re.sub(r'[^A-Za-z0-9\s.,;:()\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =============================================================================
# 2. AGENTIC CHUNKER CLASS
# =============================================================================

class AgenticChunker:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")
        self.llm = ChatOpenAI(model_name='gpt-4-turbo', openai_api_key=openai_api_key, temperature=0)

    def add_propositions(self, propositions: List[str]):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition: str):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return
        chunk_id = self._find_relevant_chunk(proposition)
        if chunk_id:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("No chunks found, creating a new one")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id: str, proposition: str):
        self.chunks[chunk_id]['propositions'].append(proposition)
        if self.generate_new_metadata_ind:
            joined_props = "\n".join(self.chunks[chunk_id]['propositions'])
            truncated_props = truncate_text(joined_props, 1000)
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(truncated_props,
                                                                          self.chunks[chunk_id].get('summary', ""))
            self.chunks[chunk_id]['title'] = self._update_chunk_title(truncated_props,
                                                                      self.chunks[chunk_id].get('title', ""))

    def _update_chunk_summary(self, propositions_text: str, current_summary: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are the steward of a group of chunks representing similar topics. Generate a very brief 1-sentence summary for this chunk. Only respond with the summary."),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}")
            ]
        )
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {
            "proposition": propositions_text,
            "current_summary": current_summary
        })
        return result.content

    def _update_chunk_title(self, propositions_text: str, current_title: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are the steward of a group of chunks representing similar topics. Generate a very brief title for this chunk. Only respond with the title."),
                ("user",
                 "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}")
            ]
        )
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {
            "proposition": propositions_text,
            "current_summary": "",
            "current_title": current_title
        })
        return result.content

    def _get_new_chunk_summary(self, proposition: str) -> str:
        truncated_prop = truncate_text(proposition, 1000)
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Generate a very brief 1-sentence summary for a new chunk based on the proposition. Only respond with the summary."),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}")
            ]
        )
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {"proposition": truncated_prop})
        return result.content

    def _get_new_chunk_title(self, summary: str) -> str:
        truncated_summary = truncate_text(summary, 500)
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                "system", "Generate a very brief title for a chunk based on its summary. Only respond with the title."),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}")
            ]
        )
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {"summary": truncated_summary})
        return result.content

    def _create_new_chunk(self, proposition: str):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)
        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self) -> str:
        chunk_outline = ""
        for chunk in self.chunks.values():
            chunk_outline += f"Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"
        return chunk_outline

    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        current_chunk_outline = self.get_chunk_outline()
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Determine if the following proposition should belong to one of the existing chunks. If yes, return the chunk id; if not, return 'No chunks'."),
                ("user",
                 "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user",
                 "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}")
            ]
        )
        runnable = PROMPT | self.llm
        result = safe_invoke(runnable.invoke, {
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        })
        chunk_found = result.content

        class ChunkID(BaseModel):
            chunk_id: Optional[str]

        extraction_chain = create_extraction_chain_pydantic(pydantic_schema=ChunkID, llm=self.llm)
        extraction_result = extraction_chain.invoke(chunk_found)
        if extraction_result and extraction_result.get("text"):
            extracted = extraction_result["text"]
            if isinstance(extracted, list) and len(extracted) > 0:
                chunk_found = extracted[0].chunk_id
            else:
                chunk_found = None
        else:
            chunk_found = None

        if not chunk_found or len(chunk_found) != self.id_truncate_limit:
            return None
        return chunk_found

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        elif get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk in self.chunks.values():
            print(f"Chunk #{chunk['chunk_index']} (ID: {chunk['chunk_id']})")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in chunk['propositions']:
                print(f"    - {prop}")
            print("\n")

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())


# =============================================================================
# 3. EMBEDDING FUNCTION
# =============================================================================

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)


def get_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()


# =============================================================================
# 4. INITIALIZE PINECONE INDEX USING THE NEW API
# =============================================================================

def init_pinecone_index(index_name: str, dimension: int):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key not set in environment variables")

    # Create a Pinecone instance using the API key.
    pc = Pinecone(api_key=pinecone_api_key)

    # List existing indexes.
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' does not exist. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',  # Adjust as needed: 'aws', 'gcp', or 'azure'
                region='us-west-2'  # Change to your region
            )
        )

    index = pc.Index(index_name)
    return index


# =============================================================================
# 5. MAIN FUNCTION: PROCESS ALL PDF BOOKS IN THE 'books' FOLDER
# =============================================================================

def main():
    pdf_folder_path = "./books"  # Folder "books" at the same level as this file
    if not os.path.isdir(pdf_folder_path):
        print(f"[red]PDF folder not found: {pdf_folder_path}[/red]")
        return

    pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not pdf_files:
        print(f"[red]No PDF files found in folder: {pdf_folder_path}[/red]")
        return

    all_propositions = []
    for pdf_file in pdf_files:
        print(f"[bold green]Processing PDF:[/bold green] {pdf_file}")
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text:
            continue
        cleaned_text = clean_text(raw_text)
        propositions = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        print(f"[bold blue]Extracted {len(propositions)} propositions from {os.path.basename(pdf_file)}.[/bold blue]")
        all_propositions.extend(propositions)

    if not all_propositions:
        print("[red]No propositions extracted from any PDFs.[/red]")
        return

    ac = AgenticChunker()
    ac.add_propositions(all_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='dict')

    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
    pinecone_index = init_pinecone_index(index_name, dimension=384)

    vectors = []
    for chunk in chunks.values():
        chunk_text = " ".join(chunk['propositions'])
        embedding = get_embedding(chunk_text)
        # Truncate the metadata text further to reduce payload size.
        truncated_chunk_text = truncate_text(chunk_text, 200)
        vector = {
            "id": chunk['chunk_id'],
            "values": embedding,
            "metadata": {
                "title": chunk['title'],
                "summary": chunk['summary'],
                "text": truncated_chunk_text
            }
        }
        vectors.append(vector)

    # Upsert vectors in batches to avoid exceeding request size limits.
    batch_upsert_vectors(pinecone_index, vectors, batch_size=10)
    print(f"[bold green]Finished upserting vectors into Pinecone index '{index_name}'.[/bold green]")


if __name__ == "__main__":
    main()