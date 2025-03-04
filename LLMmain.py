import os
import time
import getpass
from typing import List, Dict

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

# ----------------------------
# Initialize translation pipelines using Facebook's M2M100 model
# ----------------------------
from transformers import pipeline

# Use facebook/m2m100_418M for both directions.
# For Sinhala to English translation:
si_to_en = pipeline(
    "translation",
    model="facebook/m2m100_418M",
    src_lang="si",
    tgt_lang="en"
)
# For English to Sinhala translation:
en_to_si = pipeline(
    "translation",
    model="facebook/m2m100_418M",
    src_lang="en",
    tgt_lang="si"
)

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
spec = ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV", "us-east-1"))
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name in existing_indexes:
    desc = pc.describe_index(pinecone_index_name)
    if desc["dimension"] != 768:
        raise ValueError(
            f"Index '{pinecone_index_name}' exists with dimension {desc['dimension']}, but expected 768. "
            "Please delete the index or use a new index name."
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

# ----------------------------
# Initialize Semantic Encoder
# ----------------------------
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")


# ----------------------------
# Define Retrieval Function
# ----------------------------
def get_docs(query: str, top_k: int = 5) -> List[Dict]:
    """
    Encodes the query and retrieves top_k matching chunks from the Pinecone index.
    Returns a list of metadata dictionaries (including the 'text' and 'title').
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    if not matches:
        print("No matching documents found.")
        return []
    # Return full metadata for references (e.g., title, summary, text)
    return [match["metadata"] for match in matches]


# ----------------------------
# Initialize Groq Client and Define Answer Generation
# ----------------------------
os.environ["GROQ_API_KEY"] = groq_api_key
groq_client = Groq(api_key=groq_api_key)


def generate_answer(query: str, docs: List[Dict]) -> str:
    """
    Constructs a prompt using the retrieved documents as context and the user's query,
    then generates an answer using Groq's chat API with the Llama 70B model.
    If the input was in Sinhala, the answer is translated back to Sinhala.
    The final answer includes reference information.
    """
    if not docs:
        return "I'm sorry, I couldn't find any relevant information. Please consult your doctor for medical advice."

    # Prepare context text (assume 'text' is a short snippet from the chunk)
    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)

    # Prepare reference info (assume 'title' holds source info)
    references = [doc.get("title", "Unknown Source") for doc in docs]
    reference_text = "Sources: " + ", ".join(references)

    system_message = (
            "You are a compassionate and helpful medical chatbot designed for mothers. "
            "Answer questions in a friendly and supportive manner. "
            "If the answer involves medical advice, always append a disclaimer: "
            "'Disclaimer: This advice is informational only and is not a substitute for professional medical advice. Please contact your doctor for personalized guidance.'\n\n"
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

    final_answer = answer + "\n\n" + reference_text
    return final_answer


# ----------------------------
# Language Detection Utility
# ----------------------------
def is_sinhala(text: str) -> bool:
    """Check if the input text contains Sinhala Unicode characters."""
    return any('\u0D80' <= char <= '\u0DFF' for char in text)


# ----------------------------
# Chatbot Conversation Loop (Interactive Console)
# ----------------------------
def chatbot():
    print("Welcome to the MommyCare Medical Chatbot!")
    print("You can ask questions or share your feelings. To exit, type 'thank you' or 'bye'.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["thank you", "thanks", "bye"]:
            print("Chatbot: You're welcome. Take care!")
            break

        # If the query is in Sinhala, translate it to English first.
        if is_sinhala(query):
            print("[Info] Detected Sinhala input. Translating query to English...")
            translated_query = si_to_en(query)[0]['translation_text']
            print(f"[Translated Query] {translated_query}")
        else:
            translated_query = query

        docs = get_docs(translated_query, top_k=5)
        print("\n--- Retrieved Context ---")
        for doc in docs:
            print(doc.get("text", ""))
            print("---")

        english_answer = generate_answer(translated_query, docs)

        # If the original query was Sinhala, translate the answer back to Sinhala.
        if is_sinhala(query):
            print("[Info] Translating answer back to Sinhala...")
            sinhala_answer = en_to_si(english_answer)[0]['translation_text']
            final_answer = sinhala_answer
        else:
            final_answer = english_answer

        print("\nChatbot:", final_answer)
        print("\n")


if __name__ == "__main__":
    chatbot()