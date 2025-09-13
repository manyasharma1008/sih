from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_embedding_model():
    """
    Loads a small embedding model for symptom similarity.
    CPU-friendly.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_llm(model_name="google/flan-t5-small"):
    """
    Loads a small CPU-friendly LLM for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=150)
