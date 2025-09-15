import streamlit as st
from chatbot.model_loader import load_llm

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 AI Symptom Checker (LLM Only)")

# Load LLM
@st.cache_resource
def init_llm():
    return load_llm("google/flan-t5-small")

llm = init_llm()

# Prediction function using few-shot prompt
def predict(symptoms):
    symptoms_text = ", ".join(symptoms)
    prompt = f"""
You are a helpful medical assistant.
Here are examples:

Symptoms: fever, cough
Predicted Disease(s): Flu, Common Cold
Advice: Rest, drink fluids, consult doctor if persistent

Symptoms: stomach ache, nausea
Predicted Disease(s): Gastritis, Food Poisoning
Advice: Avoid heavy meals, stay hydrated, see a doctor if severe

Now, given patient symptoms: {symptoms_text}
Predict 1-3 possible diseases and give short advice.
Include disclaimer: '⚠️ This is not a medical diagnosis. Please consult a doctor.'
Format:
Predicted Disease(s): <disease names>
Advice: <short advice>
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
