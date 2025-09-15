import os
import json
import streamlit as st
import faiss
import numpy as np
from chatbot.model_loader import load_llm, load_embedding_model

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 Symptom Checker with AI")

# Load JSON symptom database
@st.cache_data
def load_symptom_dataset():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data", "symptoms.json")
    if not os.path.exists(path):
        st.warning("No local symptom database found. Will rely entirely on AI predictions.")
        return []
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            st.warning("Invalid JSON. Will rely on AI predictions.")
            return []
    return data

SYMPTOM_DB = load_symptom_dataset()

# Load models
@st.cache_resource
def init_models():
    embedder = load_embedding_model()
    llm = load_llm("google/flan-t5-small")  # CPU-friendly small LLM
    return embedder, llm

embedder, llm = init_models()

# Build FAISS index for symptom similarity
if SYMPTOM_DB:
    symptom_texts = [" ".join(entry["symptoms"]) for entry in SYMPTOM_DB]
    embeddings = embedder.encode(symptom_texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# Prediction function
def predict(symptoms):
    symptoms_lower = [s.lower() for s in symptoms]

    # 1️⃣ Exact match in JSON DB
    for entry in SYMPTOM_DB:
        if all(s in [x.lower() for x in entry["symptoms"]] for s in symptoms_lower):
            return f"**Predicted Disease(s):** {', '.join(entry['diseases'])}\n💡 Advice: {entry['advice']}\n⚠️ Not a medical diagnosis."

    # 2️⃣ FAISS similarity match
    if SYMPTOM_DB:
        query_vec = embedder.encode([" ".join(symptoms)], convert_to_numpy=True)
        D, I = index.search(query_vec, 1)
        matched = SYMPTOM_DB[I[0][0]]
        # Check similarity threshold
        if D[0][0] < 1.0:  # Adjust threshold if needed
            return f"**Predicted Disease(s):** {', '.join(matched['diseases'])}\n💡 Advice: {matched['advice']}\n⚠️ Not a medical diagnosis."

    # 3️⃣ Fallback: LLM prediction
    prompt = f"""
You are a helpful medical assistant.
Patient symptoms: {', '.join(symptoms)}

Based on these symptoms, suggest 1-3 possible diseases and 1-line advice for each.
Format like:
Predicted Disease(s): <disease names>
Advice: <short advice>
Include disclaimer: '⚠️ This is not a medical diagnosis. Please consult a doctor.'
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
                output = predict(symptoms)
            st.info(output)
    else:
        st.error("Please enter at least one symptom.")
