import os
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# For conversation context storage (in-memory; production use a database)
conversations: Dict[str, List[Dict[str, str]]] = {}

# ----------------------------
# Retrieve API Keys from .env
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env file.")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env file.")

# ----------------------------
# Initialize Pinecone Index
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-west-2")
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME in existing_indexes:
    desc = pc.describe_index(PINECONE_INDEX_NAME)
    if desc["dimension"] != 768:
        raise ValueError(
            f"Index '{PINECONE_INDEX_NAME}' exists with dimension {desc['dimension']}, but expected 768. "
            "Please delete the index via the Pinecone dashboard or use a new index name."
        )
else:
    print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating it...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec,
        deletion_protection=False
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
        time.sleep(1)
index = pc.Index(PINECONE_INDEX_NAME)
time.sleep(1)

# ----------------------------
# Initialize Semantic Encoder
# ----------------------------
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")


# ----------------------------
# Define Retrieval Function
# ----------------------------
def get_docs(query: str, top_k: int = 5) -> List[dict]:
    """
    Encodes the query and retrieves the top_k matching chunks from the Pinecone index.
    Returns a list of metadata dictionaries including keys like 'text' and 'title'.
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    if not matches:
        return []
    return [match["metadata"] for match in matches]


# ----------------------------
# Initialize Groq Client and Define Answer Generation
# ----------------------------
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
groq_client = Groq(api_key=GROQ_API_KEY)


def generate_answer(query: str, docs: List[dict]) -> str:
    """
    Constructs a prompt using the retrieved documents as context and the user's query.
    Generates an answer using Groq's chat API with the Llama 70B model, then appends a disclaimer
    and source attribution (using the 'title' field from the metadata).
    """
    if not docs:
        return "I'm sorry, I couldn't find any relevant information. Please contact your doctor for medical advice."

    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)

    references = [doc.get("title", "Unknown Source") for doc in docs]
    reference_text = "Sources: " + ", ".join(references)

    system_message = (
            "You are a compassionate and helpful medical chatbot designed for mothers. "
            "Answer questions and offer supportive advice. "
            "If your answer includes any medical advice, include a disclaimer at the end advising users to contact their doctor for personalized advice.\n\n"
            "CONTEXT:\n" + context
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    try:
        chat_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        answer = chat_response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    disclaimer = "\n\nDisclaimer: This advice is informational only and is not a substitute for professional medical advice. Please contact your doctor for personalized guidance."
    final_answer = answer + "\n\n" + reference_text + disclaimer
    return final_answer


# ----------------------------
# FastAPI Models
# ----------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    conversation: List[Dict[str, str]]  # List of messages (each with 'role' and 'content')


# ----------------------------
# Create FastAPI App
# ----------------------------
app = FastAPI(title="MommyCare Medical Chatbot Backend")


# ----------------------------
# Chat Endpoint
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message provided.")

    # Get or initialize conversation history (in-memory)
    if user_id not in conversations:
        # Initialize with a system prompt to set the assistant's role
        conversations[user_id] = [{"role": "system",
                                   "content": "You are a compassionate medical chatbot. Provide supportive and accurate responses, but always advise users to contact a doctor for personalized medical advice."}]

    # Append user message
    conversations[user_id].append({"role": "user", "content": user_message})

    # Retrieve context from Pinecone using the latest query
    docs = get_docs(user_message, top_k=5)

    # Generate answer
    answer = generate_answer(user_message, docs)

    # Append assistant's answer to conversation history
    conversations[user_id].append({"role": "assistant", "content": answer})

    # Return response along with updated conversation history
    return ChatResponse(reply=answer, conversation=conversations[user_id])


# ----------------------------
# Run FastAPI (for development)
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)