from loader import pdf_load_split
from embedder import embed_texts
from indexer import build_faiss_index,search
from generator import load_generator,generate_answer
from verifier import load_verifier,verify_answer

def main():
    
    print("Loading PDF................")
    chunks = pdf_load_split("../data/sample.pdf")
    texts = [c.page_content for c in chunks]

    print("Creating embeddings....... ")
    embeddings = embed_texts(texts)

    print("Building FAISS index ....")
    index = build_faiss_index(embeddings)

    print("Loading Microsoft PH.....")
    generator = load_generator()

    print("Loading LLamaa ............")
    verifier = load_verifier()

    print("\n Rag ready!, type exit to quit. \n")

    while True:
        q = input("Question: ")
        if q.lower() in ["exit","quit"]:
            break

        q_embed = embed_texts([q])
        ids, _ = search(index, q_embed, k = 3)
        context = "\n\n".join([texts[i] for i in ids])

        answer = generate_answer(generator,context,q)
        print("\n Ph-2 answer:\n",answer)


        verified = verify_answer(verifier,context,q,answer)
        print("\n LLama double check the answers of Mic. PH: ",verified,"\n")

if __name__ == "__main__":
    main()


