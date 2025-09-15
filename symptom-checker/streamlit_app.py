import os
import json
import streamlit as st
from chatbot.model_loader import load_llm, load_embedding_model

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 Symptom Checker with AI")

# Load symptom dataset (JSON)
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
    embedder = load_embedding_model()  # optional for FAISS or similarity search
    # Use smaller CPU-friendly model
    llm = load_llm("google/flan-t5-small")  
    return embedder, llm

embedder, llm = init_models()

# Prediction function
# Prediction function
def predict(symptoms):
    symptoms_lower = [s.lower() for s in symptoms]

    # 1️⃣ Check hardcoded/JSON database first
    for entry in SYMPTOM_DB:
        if all(s in [x.lower() for x in entry["symptoms"]] for s in symptoms_lower):
            return f"**Predicted Disease(s):** {', '.join(entry['diseases'])}\n💡 Advice: {entry['advice']}\n⚠️ Not a medical diagnosis."

    # 2️⃣ If not found, use LLM with better prompt
    prompt = f"""
You are a helpful medical assistant.
Patient symptoms: {', '.join(symptoms)}

Based on these symptoms, suggest 1-3 possible diseases and give 1-line advice for each.
Example format:
Predicted Disease(s): Common Cold, Flu
Advice: Rest, stay hydrated, consult a doctor if symptoms persist.

Do not repeat placeholders like 'disease name(s)'.
Always include: '⚠️ This is not a medical diagnosis. Please consult a doctor.'
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
