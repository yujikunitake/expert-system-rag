import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
load_dotenv()
from src.database import search_relevant_documents
from src.llm_service import generate_rag_response

print("Buscando documentos...")
docs, score = search_relevant_documents("Como funciona a politica de ferias?")
print(f"Encontrados {len(docs)} documentos. Score de busca: {score:.4f}")

print("Gerando resposta...")
try:
    resp = generate_rag_response("Como funciona a politica de ferias?", docs)
    print(resp)
except Exception as e:
    print("ERRO:", e)
    import traceback
    traceback.print_exc()
