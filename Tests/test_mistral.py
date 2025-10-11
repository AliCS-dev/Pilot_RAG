#Test for the Generator AI
import torch
from transformers import pipeline

print("Loading Mistral locally...")
model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "Tell me about Hungary:"
result = model(prompt, max_new_tokens=50)

print("Mistral response:")
print(result[0]['generated_text'])