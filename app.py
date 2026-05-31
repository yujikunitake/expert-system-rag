import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
# pyrefly: ignore [missing-import]
from src.database import search_relevant_documents, invalidate_vectorstore_cache
# pyrefly: ignore [missing-import]
from src.llm_service import generate_rag_response, reformulate_query

st.set_page_config(
    page_title="Assistente de RH Corporativo",
    layout="wide"
)

# CSS adjustments for full-screen application feel
st.markdown("""
<style>
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
footer {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

def render_assistant_message(content: str):
    if "[METADATA]" in content:
        parts = content.split("[METADATA]")
        main_text = parts[0].strip()
        metadata_text = parts[1].strip()
        
        st.markdown(main_text, unsafe_allow_html=True)
        
        citations = []
        for line in metadata_text.split("\n"):
            if "|" in line:
                subparts = line.split("|")
                if len(subparts) >= 4:
                    num = subparts[0].strip()
                    file_name = subparts[1].strip()
                    section = subparts[2].strip()
                    orig_sentence = subparts[3].strip()
                    citations.append((num, file_name, section, orig_sentence))
                elif len(subparts) == 3:
                    # Backward compatibility for 3-column legacy formats
                    num = subparts[0].strip()
                    file_name = subparts[1].strip()
                    orig_sentence = subparts[2].strip()
                    citations.append((num, file_name, "Geral", orig_sentence))
                    
        if citations:
            with st.expander("Ver fontes da informação"):
                for num, file_name, section, orig_sentence in citations:
                    st.markdown(f"**[{num}] {file_name} — {section}**")
                    st.markdown(f"*{orig_sentence}*")
                    if num != citations[-1][0]:
                        st.markdown("---")
    else:
        st.markdown(content, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
chat_enabled = api_key and api_key != "sua-chave-aqui"

col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
with col1:
    st.markdown("### Assistente de RH")
with col2:
    if st.button("Reindexar Banco", use_container_width=True, type="secondary", help="Deleta o banco vetorial atual e o reconstrói a partir dos arquivos na pasta data"):
        with st.spinner("Atualizando banco vetorial..."):
            # Lazy imports to avoid loading HuggingFace on every app boot
            import shutil
            # pyrefly: ignore [missing-import]
            from ingest import ingest_data
            db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
            ingest_data()
            invalidate_vectorstore_cache()
        st.toast("Sucesso: Banco vetorial reindexado!")
with col3:
    if st.button("Resetar Sessão", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

if not chat_enabled:
    st.warning("Chave API (OPENROUTER_API_KEY) não configurada no arquivo .env.")

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; margin-top: 4rem; margin-bottom: 2rem;">
        <h1 style="font-size: 3.2rem; font-weight: 600; background: linear-gradient(135deg, #0056b3, #007aff, #00a2e8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.8rem; font-family: 'Google Sans', Inter, sans-serif;">
            Olá, como posso ajudar?
        </h1>
        <p style="color: #5f6368; font-size: 1.2rem; font-family: 'Inter', sans-serif;">
            Pergunte-me sobre políticas de benefícios, férias, código de conduta e outros temas de Recursos Humanos.
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg["content"])
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Como funciona o processo de admissão?", disabled=not chat_enabled):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Buscando documentos...")
        
        search_query = reformulate_query(prompt, st.session_state.messages[:-1])
        retrieved_documents, best_score = search_relevant_documents(search_query)
        print(f"[RAG Flow] Original: '{prompt}' | Reformulado: '{search_query}' | Score: {best_score:.4f}")
        
        response_text = None
        
        # Threshold of 1.10 is optimized for HuggingFace all-MiniLM-L6-v2 embeddings
        if best_score <= 1.10:
            message_placeholder.markdown("Analisando contexto e gerando resposta...")
            try:
                response_text = generate_rag_response(prompt, retrieved_documents, st.session_state.messages[:-1])
            except Exception as e:
                message_placeholder.error(f"Erro ao conectar com a IA: {e}")
        else:
            greetings = ["oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "tudo bem", "como vai"]
            clean_prompt = prompt.lower().strip().replace("?", "").replace("!", "").replace(",", "").strip()
            
            is_greeting = any(clean_prompt.startswith(g) or clean_prompt == g for g in greetings)
            is_scope_query = any(k in clean_prompt for k in ["quais temas", "sobre o que", "como ajuda", "pode ajudar", "qual o seu alcance"])
            
            if is_greeting or is_scope_query:
                response_text = "Olá! Tudo bem? Como assistente de Recursos Humanos, estou aqui para ajudar você com as diretrizes da nossa empresa! Posso tirar dúvidas sobre benefícios (como férias, auxílio alimentação e refeição), licenças, onboarding (integração), dress code (código de vestimenta), uso de equipamentos, jornada de trabalho, home office e reembolso de cursos/capacitações. Como posso te ajudar hoje com esses temas?"
            else:
                response_text = "Desculpe, mas não tenho essa informação nas diretrizes oficiais de Recursos Humanos. Meu alcance é responder a perguntas exclusivamente sobre políticas de RH, benefícios, férias e admissão na nossa empresa. Como posso te ajudar hoje com esses assuntos?"
        
        if response_text:
            message_placeholder.empty()
            render_assistant_message(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
