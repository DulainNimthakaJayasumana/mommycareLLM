from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import get_docs, generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_answer/")
def get_answer(request: QueryRequest):
    try:
        docs = get_docs(request.query, top_k=5)
        answer = generate_answer(request.query, docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
