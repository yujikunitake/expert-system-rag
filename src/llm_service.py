import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_rag_llm = None
_reformulate_llm = None

def _get_rag_llm(api_key: str):
    global _rag_llm
    if _rag_llm is None:
        _rag_llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=500,
            temperature=0.3  # Temperature of 0.3 balances humanized response with factual constraints
        )
    return _rag_llm

def _get_reformulate_llm(api_key: str):
    global _reformulate_llm
    if _reformulate_llm is None:
        _reformulate_llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=100,
            temperature=0.0
        )
    return _reformulate_llm

def generate_rag_response(question: str, retrieved_documents: list, chat_history: list | None = None) -> str:
    chat_history = chat_history or []
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Erro: Chave de API da OpenRouter não configurada na interface."
    
    llm = _get_rag_llm(api_key)
    
    formatted_contexts = []
    for i, doc in enumerate(retrieved_documents):
        source = doc.metadata.get('source', 'Documento Desconhecido')
        source_name = os.path.basename(source)
        formatted_contexts.append(f"DOCUMENTO {i+1} (FONTE: {source_name}):\n{doc.page_content}")
        
    context_text = "\n\n".join(formatted_contexts)
    has_history = len(chat_history) > 0
    
    system_instruction = f"""Você é um Assistente de Recursos Humanos da Empresa. Escreva respostas de forma empática, extremamente humanizada, natural, amigável e direta.
    
REGRAS CRÍTICAS DE TOM, SAUDAÇÃO E EMOJIS:
1. SAUDAÇÃO APENAS NA PRIMEIRA MENSAGEM: Se houver histórico de conversa anterior, você NÃO DEVE incluir NENHUMA saudação (como "Olá", "Oi", "Tudo bem?") ou frase de transição/boas-vindas (como "Que bom que voltou a conversar", "Fico feliz em ajudar de novo", "Olá novamente", etc.). Responda à pergunta atual de forma direta e profissional, mantendo o tom empático e humanizado, mas sem saudações.
   *(Nota de contexto para esta resposta específica: {'NÃO inclua nenhuma saudação' if has_history else 'você pode incluir uma saudação inicial amigável como primeiro contato'}).*
2. ZERO EMOJIS: Não utilize emojis em nenhuma parte do seu texto para manter a comunicação visualmente limpa e profissional.
3. Quando questionado sobre políticas e procedimentos da empresa, você DEVE basear-se ESTRITAMENTE no contexto fornecido na última mensagem.

REGRAS CRÍTICAS DE DESTAQUE E RASTREABILIDADE (CITAÇÕES):
1. ZERO ALUCINAÇÃO SOBRE REGRAS: Jamais invente ou suponha regras de RH.
2. FORA DO ESCOPO: Se a pergunta for sobre um assunto de RH que não está no contexto, diga gentilmente que não possui essa informação nas diretrizes oficiais e recomende falar com o gestor.
3. DESTAQUE (GRIFO) INDIVIDUAL DE CADA FATO: Cada fato, regra, período de antecedência, prazo ou dado específico que você extrair do contexto DEVE ser colocado em negrito usando `**texto**` e receber um número de citação ÚNICO entre colchetes (ex: [1], [2], [3], etc.).
   ATENÇÃO MÁXIMA: NÃO junte múltiplas afirmações ou frases distintas sob o mesmo número de citação. Cada afirmação ou frase que contenha uma informação factual deve ter sua própria citação separada e sequencial.
   Exemplo incorreto: "...período de 12 meses [1]. A programação deve ser feita com 45 dias [1]."
   Exemplo correto: "...período de 12 meses [1]. A programação deve ser feita com 45 dias [2]."
4. BLOCO DE METADADOS: No final absoluto do seu texto, adicione uma linha contendo exatamente a palavra [METADATA] (tudo em maiúsculas). Abaixo dessa linha, para CADA citação numérica gerada no texto ([1], [2], [3], etc.), adicione OBRIGATORIAMENTE uma linha correspondente com o formato:
   [Número da citação] | [Nome do Arquivo] | [Nome do Subtítulo ou Seção] | [Frase Original exata extraída do contexto]
   
   Identifique o subtítulo ou a seção correspondente no contexto (ex: "1. Diretrizes de Férias", "3. Dress Code", etc.).
   
   Exemplo do formato final completo de saída:
   O período aquisitivo de férias é de **12 meses** [1]. Lembre-se de que a programação de férias deve ser feita com **45 dias de antecedência** [2].
   
   [METADATA]
   1 | manual_beneficios_ferias.txt | 1. Diretrizes de Férias | O período aquisitivo de férias é de 12 meses.
   2 | manual_beneficios_ferias.txt | 1. Diretrizes de Férias | A programação de férias deve ser solicitada via sistema de RH (SoftExpert) com 45 dias de antecedência.
   
   ATENÇÃO: Cada número citado no texto ([1], [2], etc.) deve possuir exatamente UMA linha correspondente abaixo de [METADATA]. Mantenha rigorosamente o caractere "|" separando os campos no bloco de metadados. Não use formatações adicionais (como negrito ou itálico) no bloco de metadados."""

    messages = [SystemMessage(content=system_instruction)]
    
    # We deliberately preserve [METADATA] in history to serve as a perfect few-shot in-context learning anchor.
    for msg in chat_history:
        content = msg["content"]
        if msg["role"] == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
            
    user_prompt = f"""CONTEXTO RECUPERADO DA BASE:
{context_text}

PERGUNTA ATUAL DO COLABORADOR:
{question}

ATENÇÃO IMPORTANTE: Lembre-se de grifar individualmente em negrito cada fato da resposta e adicionar obrigatoriamente o bloco [METADATA] ao final absoluto do seu texto, com as quatro colunas separadas por '|' para cada citação numérica gerada, conforme as regras do prompt de sistema."""

    messages.append(HumanMessage(content=user_prompt))
    
    response = llm.invoke(messages)
    return response.content

def reformulate_query(question: str, chat_history: list | None = None) -> str:
    chat_history = chat_history or []
    # Short-circuit if no history is present (saves API latency and token cost)
    if not chat_history:
        return question
        
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return question
    
    try:
        llm = _get_reformulate_llm(api_key)
        
        system_instruction = """Dada uma conversa de chat e uma nova pergunta do colaborador, reescreva a pergunta para que ela seja uma pergunta autônoma (self-contained), em português, que possa ser buscada de forma independente em um banco de dados de documentos de RH.
Não adicione explicações, responda APENAS com a pergunta reescrita. Se a pergunta já for autônoma ou se não houver contexto suficiente para reformular, retorne a pergunta original exatamente como ela é."""
        
        messages = [SystemMessage(content=system_instruction)]
        
        # Limit history context to the last 4 turns to optimize throughput and minimize cost
        for msg in chat_history[-4:]:
            content = msg["content"]
            if "[METADATA]" in content:
                content = content.split("[METADATA]")[0].strip()
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
                
        messages.append(HumanMessage(content=f"Nova Pergunta: {question}"))
        
        response = llm.invoke(messages)
        reformulated = response.content.strip()
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1].strip()
        return reformulated
    except Exception:
        # Fallback to the original question if API request fails to avoid blocking the user
        return question
