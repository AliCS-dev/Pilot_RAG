from transformers import AutoTokenizer, AutoModelForCausalLM, pipelines, BitsAndBytesConfig

def load_generator():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    # Create quantization configuration for 4-bit model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,  # use new argument instead of load_in_4bit
        device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True  # allows CPU offloading for large layers
    )

    pipe = pipelines("text-generation", model=model, tokenizer=tokenizer)
    return pipe

def generate_answer(pipe, context, question):
    prompt = f"Use the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    out = pipe(prompt, max_new_tokens=150, do_sample=False)
    return out[0]["generated_text"]