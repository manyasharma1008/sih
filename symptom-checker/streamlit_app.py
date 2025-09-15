import os
import json
import streamlit as st
from chatbot.model_loader import load_llm, load_embedding_model

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 Symptom Checker with AI")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "data", "symptoms.json")
    if not os.path.exists(json_path):
        st.error(f"Cannot find symptoms.json at {json_path}")
        return {}
    with open(json_path, "r") as f:
        try:
            symptoms_dict = json.load(f)
        except json.JSONDecodeError:
            st.error("symptoms.json is not a valid JSON file.")
            return {}
    return symptoms_dict

symptoms_dict = load_data()
if not symptoms_dict:
    st.stop()  

@st.cache_resource
def init_models():
    embedder = load_embedding_model()  
    llm = load_llm("google/flan-t5-small")
    return embedder, llm

embedder, llm = init_models()

SYMPTOM_DB = [
    {"symptoms": ["fever", "cough", "sore throat"], 
     "diseases": ["Common Cold", "Flu", "COVID-19"], 
     "advice": "Stay hydrated, rest, and consult a doctor if symptoms persist more than 3 days."},
    {"symptoms": ["chest pain", "shortness of breath", "dizziness"], 
     "diseases": ["Heart Attack", "Angina", "Panic Attack"], 
     "advice": "Seek immediate medical help. Call emergency services."},
    {"symptoms": ["headache", "sensitivity to light", "stiff neck"], 
     "diseases": ["Migraine", "Meningitis"], 
     "advice": "If sudden and severe, seek emergency medical attention."}
]

# Prediction function
def predict(symptoms):
    symptoms_lower = [s.lower() for s in symptoms]

    # Check hardcoded symptoms
    for entry in SYMPTOM_DB:
        if all(s in [x.lower() for x in entry["symptoms"]] for s in symptoms_lower):
            return f"**Predicted Disease(s):** {', '.join(entry['diseases'])}\n💡 Advice: {entry['advice']}"

    # No exact match -> feed directly to LLM
    prompt = f"""
Patient symptoms: {', '.join(symptoms)}
Please suggest possible diseases and precautions in 2-3 lines.
Include a disclaimer: 'This is not a medical diagnosis. Please consult a doctor.'
"""
    result = llm(prompt)
    return result[0]["generated_text"].strip()

# Streamlit UI
user_input = st.text_input("Enter your symptoms (comma-separated):")
if st.button("Predict"):
    if user_input:
        symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
        if not symptoms:
            st.error("Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                result = predict(symptoms)
            st.info(result)
    else:
        st.error("Please enter at least one symptom.")
