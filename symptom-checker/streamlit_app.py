import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 AI Symptom Checker (LLM Only)")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"  # CPU-friendly, bigger than small
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # CPU or GPU if available
        max_new_tokens=150
    )
    return llm_pipeline

llm = load_llm()

# ---------------------------
# Prediction function
# ---------------------------
def predict(symptoms):
    # Few-shot prompt for better disease prediction
    prompt = f"""
You are a professional medical assistant.

Predict 1-3 possible diseases from the given symptoms and provide one-line advice for each. 
Include this disclaimer at the end: '⚠️ This is not a medical diagnosis. Please consult a doctor.'

Examples:
Symptoms: fever, cough
Predicted Disease(s): Common Cold, Flu
Advice: Stay hydrated, rest, and consult a doctor if symptoms persist.

Symptoms: stomach ache, nausea
Predicted Disease(s): Gastritis, Food Poisoning
Advice: Avoid heavy meals, drink water, and see a doctor if severe.

Patient Symptoms: {', '.join(symptoms)}
Predicted Disease(s):
Advice:
"""
    result = llm(prompt)
    return result[0]["generated_text"].strip()

# ---------------------------
# Streamlit UI
# ---------------------------
user_input = st.text_input("Enter your symptoms (comma-separated):")
if st.button("Predict"):
    if user_input:
        symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
        if not symptoms:
            st.error("Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                output = predict(symptoms)
            st.success(output)
    else:
        st.error("Please enter at least one symptom.")
