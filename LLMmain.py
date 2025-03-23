import os
import time
import uuid
import re
import glob
from typing import Optional, List
import asyncio

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Pinecone client using the new API:
from pinecone import Pinecone, ServerlessSpec

# Semantic encoder from semantic_router
from semantic_router.encoders import HuggingFaceEncoder

# Groq client for Llama 70B generation
from groq import Groq

# Retrieve API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env file.")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-west-2")
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name in existing_indexes:
    desc = pc.describe_index(pinecone_index_name)
    if desc["dimension"] != 768:
        raise ValueError(
            f"Index '{pinecone_index_name}' exists with dimension {desc['dimension']}, but expected 768."
        )
else:
    print(f"Index '{pinecone_index_name}' does not exist. Creating it...")
    pc.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=spec,
        deletion_protection=False
    )
    while not pc.describe_index(pinecone_index_name).status.get("ready", False):
        time.sleep(1)
index = pc.Index(pinecone_index_name)
time.sleep(1)

# Initialize Semantic Encoder
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Initialize Groq Client
os.environ["GROQ_API_KEY"] = groq_api_key
groq_client = Groq(api_key=groq_api_key)

def get_docs(query: str, top_k: int = 5) -> List[dict]:
    """
    Encodes the query and retrieves top_k matching chunks from the Pinecone index.
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    if not matches:
        print("[red]No matching documents found.[/red]")
        return []
    return [match["metadata"] for match in matches]

# Agentic Chunker Class with reduced chunk size
class AgenticChunker:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.id_truncate_limit = 5
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY not provided in environment variables.")
        from langchain_community.chat_models.openai import ChatOpenAI  # Ensure correct import
        self.llm = ChatOpenAI(model_name='gpt-4-turbo', openai_api_key=openai_api_key, temperature=0.2)

    def chunk_document(self, text: str, target_chars: int = 1500, overlap_chars: int = 150) -> List[dict]:
        """
        Splits text into overlapping chunks; using smaller sizes to reduce resource load.
        """
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
        truncated_text = text if len(text) <= 1000 else text[:1000] + "..."
        PROMPT = self.llm.model.build_prompt([
            ("system", "Generate a concise 1-sentence summary of the following text chunk."),
            ("user", f"Text chunk:\n{text}")
        ])
        result = self.llm(prompt=PROMPT)
        return result.strip()

    def _get_new_chunk_title(self, summary: str) -> str:
        truncated_summary = summary if len(summary) <= 500 else summary[:500] + "..."
        PROMPT = self.llm.model.build_prompt([
            ("system", "Generate a brief title capturing the main topic from the summary."),
            ("user", f"Summary:\n{truncated_summary}")
        ])
        result = self.llm(prompt=PROMPT)
        return result.strip()

# Embedding function using transformers
from transformers import AutoTokenizer, AutoModel
import torch
EMBEDDING_MODEL_NAME = "dwzhu/e5-base-4k"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

def get_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.tolist()

# Async answer generation with timeout
async def generate_answer_async(query: str, docs: List[dict]) -> str:
    if not docs:
        return "I'm sorry, I couldn't find any relevant information."
    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)
    system_message = (
        "You are a compassionate and helpful medical chatbot designed for mothers. "
        "Answer the question clearly and concisely based solely on the provided context.\n\n"
        "CONTEXT:\n" + context
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    try:
        response = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages
            ),
            timeout=30
        )
        answer = response.choices[0].message.content
    except asyncio.TimeoutError:
        answer = "The request timed out. Please try again later."
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    return answer

def generate_answer(query: str, docs: List[dict]) -> str:
    return asyncio.run(generate_answer_async(query, docs))

def chatbot():
    print("Welcome to the MommyCare Medical Chatbot!")
    print("You can ask any questions or share your feelings. Type 'thank you' or 'bye' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["thank you", "thanks", "bye"]:
            print("Chatbot: You're welcome. Take care!")
            break
        docs = get_docs(query, top_k=5)
        print("\n--- Retrieved Context ---")
        for doc in docs:
            print(doc.get("text", ""))
            print("---")
        answer = generate_answer(query, docs)
        print("\nChatbot:", answer)
        print("\n")

if __name__ == "__main__":
    chatbot()