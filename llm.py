import os
import getpass
import time
from typing import List

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Pinecone client using the new API:
from pinecone import Pinecone, ServerlessSpec

# Semantic encoder from semantic_router
from semantic_router.encoders import HuggingFaceEncoder

# Groq client for Llama 70B generation
from groq import Groq

# ----------------------------
# Retrieve API Keys from .env
# ----------------------------
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env file.")

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in .env file.")

# ----------------------------
# Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-west-2")
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name in existing_indexes:
    desc = pc.describe_index(pinecone_index_name)
    if desc["dimension"] != 768:
        raise ValueError(f"Index '{pinecone_index_name}' exists with dimension {desc['dimension']}, but expected 768.")
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

# ----------------------------
# Initialize Semantic Encoder
# ----------------------------
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# ----------------------------
# Define Retrieval Function
# ----------------------------
def get_docs(query: str, top_k: int = 5) -> List[str]:
    """
    Encodes the query and retrieves top_k matching chunks from the Pinecone index.
    Returns a list of the 'text' fields from the metadata.
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    docs = [match["metadata"]["text"] for match in res["matches"]]
    return docs

# ----------------------------
# Initialize Groq Client and Define Answer Generation
# ----------------------------
os.environ["GROQ_API_KEY"] = groq_api_key
groq_client = Groq(api_key=groq_api_key)

def generate_answer(query: str, docs: List[str]) -> str:
    """
    Constructs a prompt using the retrieved documents as context and the user's query,
    then generates an answer using Groq's chat API with the Llama 70B model.
    """
    system_message = (
        "You are a helpful medical assistant that answers questions based on the provided context. "
        "If the answer is not clearly in the context, say 'I don't know'.\n\n"
        "CONTEXT:\n" + "\n---\n".join(docs)
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    chat_response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages
    )
    return chat_response.choices[0].message.content

# ----------------------------
# Main: Run the Q&A Pipeline
# ----------------------------
if __name__ == "__main__":
    query = input("Enter your medical question: ")
    docs = get_docs(query, top_k=5)
    print("\n--- Retrieved Context ---")
    for doc in docs:
        print(doc)
        print("---")
    answer = generate_answer(query, docs)
    print("\n--- Answer ---")
    print(answer)