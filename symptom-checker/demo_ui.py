import gradio as gr
from chatbot.rag_pipeline import SymptomChatbot

bot = SymptomChatbot(model_name="google/flan-t5-small")

def chatbot_fn(symptoms_text):
    symptoms = [s.strip().lower() for s in symptoms_text.split(",") if s.strip()]
    return bot.get_response(symptoms)

demo = gr.Interface(
    fn=chatbot_fn,
    inputs=gr.Textbox(label="Enter your symptoms (comma-separated)"),
    outputs=gr.Textbox(label="Chatbot Reply"),
    title="Medical Symptom Checker Chatbot",
    description="Enter symptoms like 'fever, cough, sore throat'. The chatbot suggests possible conditions and advice. ⚠️ Not a medical diagnosis."
)

if __name__ == "__main__":
    demo.launch(share=True)
