import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from LLaMA3 import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_response(request: Request):
    data = await request.json()
    question = data.get("question")

    if not question or not question.strip():
        return {"answer": "❌ 질문이 비어 있습니다."}

    try:
        answer = generate_answer(question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"⚠️ 오류 발생: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7001)
