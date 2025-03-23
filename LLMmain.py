import os
import time
import uuid
import re
import glob
from typing import Optional, List
import asyncio

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Pinecone client using the new API:
# ----------------------------
from pinecone import Pinecone, ServerlessSpec

# ----------------------------
# Semantic encoder from semantic_router
# ----------------------------
from semantic_router.encoders import HuggingFaceEncoder

# ----------------------------
# Groq client for Llama 70B generation
# ----------------------------
from groq import Groq

# ----------------------------
# Retrieve API Keys from environment
# ----------------------------
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env file.")

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in .env file.")

# ----------------------------
# Initialize Pinecone Index
# ----------------------------
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-west-2")
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name in existing_indexes:
    desc = pc.describe_index(pinecone_index_name)
    if desc["dimension"] != 768:
        raise ValueError(
            f"Index '{pinecone_index_name}' exists with dimension {desc['dimension']}, but expected 768. Please delete the index or use a new name."
        )
else:
    print(f"Index '{pinecone_index_name}' does not exist. Creating it...")
    # noinspection PyTypeChecker
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

# ----------------------------
# Initialize Semantic Encoder
# ----------------------------
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# ----------------------------
# Initialize Groq Client
# ----------------------------
os.environ["GROQ_API_KEY"] = groq_api_key
groq_client = Groq(api_key=groq_api_key)


# ----------------------------
# Retrieval Function: get_docs
# ----------------------------
def get_docs(query: str, top_k: int = 5) -> List[dict]:
    """
    Encodes the query and retrieves top_k matching chunks from the Pinecone index.
    Returns a list of metadata dictionaries (including the 'text' and 'title').
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    if not matches:
        print("[red]No matching documents found.[/red]")
        return []
    return [match["metadata"] for match in matches]


# ----------------------------
# Answer Generation using Groq (async version)
# ----------------------------
async def generate_answer_async(query: str, docs: List[dict]) -> str:
    """
    Constructs a prompt using the retrieved documents as context and the user's query,
    then generates an answer using Groq's chat API with a 30-second timeout.
    """
    if not docs:
        return "I'm sorry, I couldn't find any relevant information."

    # Build the context from the 'text' snippets in docs
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
        # noinspection PyTypeChecker
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
    """Synchronous wrapper to run the async generate_answer_async."""
    return asyncio.run(generate_answer_async(query, docs))


# ----------------------------
# Chatbot Interactive Loop (for local testing)
# ----------------------------
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