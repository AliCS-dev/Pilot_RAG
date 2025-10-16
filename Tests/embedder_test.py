import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../src")))

from embedder import embed_texts

texts = ["I love AI","AI is the future"]

embeddings = embed_texts(texts)

print("Embedding success.")
print("Shape: ",embeddings.shape)
print("Vectors: ",embeddings[0][:30])