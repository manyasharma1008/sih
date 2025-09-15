import os
import json
import streamlit as st
import faiss
import numpy as np
from chatbot.model_loader import load_llm, load_embedding_model

# --- Streamlit page config ---
st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 Symptom Checker with AI")

# --- Load symptoms JSON ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "data", "symptoms.json")  # correct path
    if not os.path.exists(json_path):
        st.error(f"Cannot find symptoms.json at {json_path}")
        return {}  # return empty dict to avoid crash
    with open(json_path, "r") as f:
        symptoms_dict = json.load(f)
    return symptoms_dict

symptoms_dict = load_data()

# Stop app if file missing
if not symptoms_dict:
    st.stop()

all_symptoms = list(symptoms_dict.keys())

# --- Initialize models ---
@st.cache_resource
def init_models():
    embedder = load_embedding_model()  
    llm = load_llm("google/flan-t5-small")  
    return embedder, llm

embedder, llm = init_models()

# --- Hardcoded symptom DB for exact matches ---
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

# --- FAISS index for similarity search ---
symptom_texts = [" ".join(item["symptoms"]) for item in SYMPTOM_DB]
embeddings = embedder.encode(symptom_texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# --- Prediction function ---
def predict(symptoms):
    symptoms_lower = [s.lower() for s in symptoms]

    # First check hardcoded DB
    for entry in SYMPTOM_DB:
        if all(s in [x.lower() for x in entry["symptoms"]] for s in symptoms_lower):
            return f"**Predicted Disease(s):** {', '.join(entry['diseases'])}\n💡 Advice: {entry['advice']}"

    # Else use embedding + LLM
    query_vec = embedder.encode([" ".join(symptoms)], convert_to_numpy=True)
    D, I = index.search(query_vec, 1)
    matched = SYMPTOM_DB[I[0][0]]

    prompt = f"""
Patient symptoms: {', '.join(symptoms)}
Possible conditions: {', '.join(matched['diseases'])}
Suggested advice: {matched['advice']}

Please give a clear, empathetic answer with a disclaimer:
'This is not a medical diagnosis. Please consult a doctor.'
"""
    result = llm(prompt)
    return result[0]["generated_text"].strip()

# --- Streamlit UI ---
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
