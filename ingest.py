from rag_utils import load_kb, build_chroma

if __name__ == "__main__":
    kb = load_kb("db/medical_kb.csv")
    build_chroma(kb)
    print("Chroma index built.")
