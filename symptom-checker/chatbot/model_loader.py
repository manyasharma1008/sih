from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_llm(model_name="microsoft/phi-3-mini-4k-instruct"):
    """
    Load a CPU/GPU-friendly LLM for disease prediction.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # GPU if available
        max_new_tokens=150
    )
    return llm_pipeline
