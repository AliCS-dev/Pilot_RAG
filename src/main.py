import time
from loader import pdf_load_split
from embedder import embed_texts
from indexer import build_faiss_index,search
from generator import load_generator,generate_answer
from verifier import load_verifier,verify_answer

def main():
    total_start = time.time()
    
    print("Loading PDF................")
    chunks = pdf_load_split("../data/sample.pdf")
    texts = [c.page_content for c in chunks]

    start = time.time()  #checkpoint 
    print("Creating embeddings....... ")
    embeddings = embed_texts(texts)

    start = time.time() #Chek point
    print("Building FAISS index ....")
    index = build_faiss_index(embeddings)

    start = time.time() 
    print("Loading Microsoft PH.....")
    generator = load_generator()

    start = time.time() #check point
    print("Loading LLamaa ............")
    verifier = load_verifier()

    print("\n Rag ready!, type exit to quit. \n")

    while True:
        q = input("Question: ")
        if q.lower() in ["exit","quit"]:
            break

        start = time.time()

        q_embed = embed_texts([q])
        ids, _ = search(index, q_embed, k = 3)
        context = "\n\n".join([texts[i] for i in ids])

        answer = generate_answer(generator,context,q)
        print("\n Ph-2 answer:\n",answer)


        verified = verify_answer(verifier,context,q,answer)
        print("\n LLama double check the answers of Mic. PH: ",verified,"\n")

        print(f"\nResponse generated in {round(time.time() - start, 2)} sec\n")

    print(f"\nTotal runtime: {round(time.time() - total_start, 2)} sec")

if __name__ == "__main__":
    main()


