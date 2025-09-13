import os
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.rag_pipeline import SymptomChatbot
import uvicorn

app = FastAPI()
bot = SymptomChatbot()  

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "FastAPI CPU-friendly LLM Symptom Checker running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_input = request.query.lower().strip()
    
    # Keyword matching first
    matched_conditions = []
    for entry in bot.data:
        if any(symptom in user_input for symptom in entry["symptoms"]):
            matched_conditions.append(entry)
    
    if matched_conditions:
        response_list = []
        for cond in matched_conditions:
            response_list.append({
                "possible_diseases": cond["diseases"],
                "advice": cond["advice"]
            })
        return {"response": response_list}

    user_symptoms = [s.strip() for s in user_input.split(",")]
    llm_response = bot.get_response(user_symptoms)
    return {"response": llm_response}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
