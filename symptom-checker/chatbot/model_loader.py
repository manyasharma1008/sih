# chatbot/model_loader.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_llm(model_name="google/flan-t5-small"):
    """
    Load a small LLM for symptom-based disease prediction.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
        max_new_tokens=150
    )
    return llm_pipeline


from sentence_transformers import SentenceTransformer

def load_embedding_model():
    """
    Optional embedding model for symptom similarity (FAISS).
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
