import os
import uuid
import re
import glob
from typing import Optional, List

# PDF extraction
from pdfminer.high_level import extract_text

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# For pretty printing
from rich import print

# Pinecone for vector database
import pinecone

# Hugging Face for embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# LangChain imports
from langchain.prompts import ChatPromptTemplate
# Fix: Import ChatOpenAI directly from its module to avoid dependency on langchain_community
from langchain.chat_models.openai import ChatOpenAI
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic


# =============================================================================
# 1. FUNCTIONS FOR PDF EXTRACTION & TEXT CLEANING
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"[red]Error extracting text from {pdf_path}: {e}[/red]")
        return ""


def clean_text(text: str) -> str:
    """
    Cleans text by removing unwanted characters and normalizing whitespace.
    """
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

        # Whether or not to update/refine summaries and titles as new info is added
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)

    def add_propositions(self, propositions: List[str]):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition: str):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        # If no chunks exist, create a new one
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
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk: dict) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are the steward of a group of chunks representing similar topics. Generate a very brief 1-sentence summary for this chunk. Only respond with the summary."),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}")
            ]
        )
        runnable = PROMPT | self.llm
        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        }).content
        return new_chunk_summary

    def _update_chunk_title(self, chunk: dict) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are the steward of a group of chunks representing similar topics. Generate a very brief title for this chunk. Only respond with the title."),
                ("user",
                 "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}")
            ]
        )
        runnable = PROMPT | self.llm
        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content
        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Generate a very brief 1-sentence summary for a new chunk based on the proposition. Only respond with the summary."),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}")
            ]
        )
        runnable = PROMPT | self.llm
        new_chunk_summary = runnable.invoke({"proposition": proposition}).content
        return new_chunk_summary

    def _get_new_chunk_title(self, summary: str) -> str:
        PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Generate a very brief title for a chunk based on its summary. Only respond with the title."),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}")
            ]
        )
        runnable = PROMPT | self.llm
        new_chunk_title = runnable.invoke({"summary": summary}).content
        return new_chunk_title

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
        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content

        # Define a simple Pydantic model to extract the chunk id
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
# 4. INITIALIZE PINECONE INDEX
# =============================================================================

def init_pinecone_index(index_name: str, dimension: int):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("Pinecone API key or environment not set in environment variables")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if index_name not in pinecone.list_indexes():
        print(f"Index '{index_name}' does not exist. Creating new index...")
        pinecone.create_index(index_name, dimension=dimension)
    index = pinecone.Index(index_name)
    return index


# =============================================================================
# 5. MAIN FUNCTION: PROCESS ALL PDF BOOKS IN THE 'books' FOLDER
# =============================================================================

def main():
    # Folder containing your PDF books (folder "books" at the same level as this file)
    pdf_folder_path = "./books"
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
        # Split text into propositions based on double newlines (adjust if needed)
        propositions = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        print(f"[bold blue]Extracted {len(propositions)} propositions from {os.path.basename(pdf_file)}.[/bold blue]")
        all_propositions.extend(propositions)

    if not all_propositions:
        print("[red]No propositions extracted from any PDFs.[/red]")
        return

    # Initialize AgenticChunker and add propositions
    ac = AgenticChunker()
    ac.add_propositions(all_propositions)

    # Optionally, print out your chunks
    ac.pretty_print_chunks()

    # Get chunks as a dictionary
    chunks = ac.get_chunks(get_type='dict')

    # Initialize Pinecone index
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
    pinecone_index = init_pinecone_index(index_name, dimension=384)

    # Prepare vectors for upsertion
    vectors = []
    for chunk in chunks.values():
        chunk_text = " ".join(chunk['propositions'])
        embedding = get_embedding(chunk_text)
        vector = {
            "id": chunk['chunk_id'],
            "values": embedding,
            "metadata": {
                "title": chunk['title'],
                "summary": chunk['summary'],
                "text": chunk_text
            }
        }
        vectors.append(vector)

    # Upsert vectors into Pinecone
    upsert_response = pinecone_index.upsert(vectors=vectors)
    print(f"[bold green]Upserted {len(vectors)} chunks into Pinecone index '{index_name}'.[/bold green]")
    print("Response:", upsert_response)


if __name__ == "__main__":
    main()