import streamlit as st
from chatbot.rag_pipeline import SymptomChatbot

st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="wide")
st.title("🧠 AI Symptom Checker (LLM Only)")

# Initialize chatbot
@st.cache_resource
def init_bot():
    return SymptomChatbot(model_name="google/flan-t5-base")

bot = init_bot()

# Streamlit UI
user_input = st.text_input("Enter your symptoms (comma-separated):")

if st.button("Predict"):
    if user_input:
        symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
        if not symptoms:
            st.error("Please enter at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                output = bot.get_response(symptoms)
            st.info(output)
    else:
        st.error("Please enter at least one symptom.")
