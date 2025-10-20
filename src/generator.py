from transformers import AutoTokenizer, AutoModelForCausalLM, pipelines

def load_generator():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "auto",
        load_in_4bit = True
    )

    pipe = pipelines("text-generation", model = model , tokenizer = tokenizer)

    return pipe


def generate_answer(pipe, context, question):
    prompt = f"Use the context to answer the question. \n\nContext: \n{context}\n\nQuestion: {question}\nAnswer:"
    out = pipe(prompt, max_new_tokens = 150, do_sample= False)
    return out[0]["generated_text"]q