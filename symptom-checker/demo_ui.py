import gradio as gr
from chatbot.rag_pipeline import SymptomChatbot

bot = SymptomChatbot(model_name="microsoft/phi-3-mini-4k-instruct")

def chatbot_fn(symptoms_text):
    symptoms = [s.strip() for s in symptoms_text.split(",") if s.strip()]
    return bot.get_response(symptoms)

demo = gr.Interface(
    fn=chatbot_fn,
    inputs=gr.Textbox(label="Enter your symptoms (comma-separated)"),
    outputs=gr.Textbox(label="Chatbot Reply"),
    title="Medical Symptom Checker Chatbot",
    description="Enter any symptoms like 'fever, cough, stomach ache'. The AI predicts possible diseases and advice ⚠️ Not a medical diagnosis."
)

if __name__ == "__main__":
    demo.launch(share=True)
