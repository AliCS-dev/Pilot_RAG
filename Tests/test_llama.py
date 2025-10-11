#Test for verifier
import sys
import os
sys.path.append('../src')
from config import Huggin_face_token
from huggingface_hub import InferenceClient

def verify_answer(answer):
    client = InferenceClient(
        provider="featherless-ai",
        api_key=Huggin_face_token,
    )
    
    prompt = f"Verify if this answer sounds accurate: '{answer}'"
    result = client.text_generation(
        prompt,
        model="meta-llama/Meta-Llama-3-8B",
        max_new_tokens=50
    )
    return result