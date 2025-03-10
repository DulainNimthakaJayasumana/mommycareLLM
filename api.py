from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from LLMmain import get_docs, generate_answer
from trans import sinhalaToEnglish, englishToSinhala

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_answer/")
def get_answer(request: QueryRequest):
    try:
        english_query = sinhalaToEnglish(request.query)

        docs = get_docs(english_query, top_k=5)
        answer = generate_answer(request.query, docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_sinhala/")
def get_answer_sinhala(request: QueryRequest):
    try:
        # First translate Sinhala to English
        english_query = sinhalaToEnglish(request.query)      
        # Get documents based on the English query
        docs = get_docs(english_query, top_k=5)       
        # Generate answer in English
        english_answer = generate_answer(english_query, docs)
        # Translate the answer back to Sinhala
        sinhala_answer = englishToSinhala(english_answer)
        
        return {"answer": sinhala_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)