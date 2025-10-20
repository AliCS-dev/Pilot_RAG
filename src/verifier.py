from huggingface_hub import InferenceClient
from config import Huggin_face_token

def load_verifier():
    client = InferenceClient(
        provider="featherless-ai",
        api_key= Huggin_face_token,
    )
    return client

def verify_answer(client, context,question,answer):
    prompt = f"""
You are a fact checker. Verify the accuracy of the ANSWER using the CONTEXT.
If it is incorrect, fix it. If unsure, say "Idk"

CONTEXT:
{context}
QUESTION:
{question}
ANSWER:
{answer}

Verified Answer:
"""
    result = client.text_generation(
        prompt,
        model = "meta-llama/Meta-Llama-3-8B",
        max_new_tokens = 150
    )

    return result