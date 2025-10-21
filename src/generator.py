from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_generator():
    

    model_name = "microsoft/phi-2"     

    print(f"Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Local generator ready.")
    return pipe


def generate_answer(pipe, context, question):
    prompt = f"""You are a helpful AI assistant.
Use the CONTEXT to answer the QUESTION as accurately as possible.

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""
    out = pipe(prompt, max_new_tokens=60, do_sample=False, temperature = 0.3, truncation = True)
    return out[0]["generated_text"]