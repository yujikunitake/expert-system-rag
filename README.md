# 💼 Assistente de Recursos Humanos (RAG)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-%E2%89%A51.35-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/chromadb-local-blueviolet.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Um agente conversacional de alto desempenho especializado em **Recursos Humanos**, implementado com a arquitetura **RAG (Retrieval-Augmented Generation)**. O sistema atua de forma genérica sobre diretrizes corporativas (férias, benefícios, onboarding e admissão), garantindo respostas humanas, precisas e totalmente auditáveis, com foco extremo em contenção de escopo e eficiência de custo de API.

---

## 🌟 Funcionalidades Principais

*   **Roteamento Semântico Inteligente (Guardrail Matemático - Zero Custo)**: O sistema analisa a distância vetorial (*L2 score*) das buscas do ChromaDB. Se a pergunta estiver fora do escopo de RH (ex: cálculos matemáticos, códigos ou conhecimentos gerais) ou for uma saudação curta, o chatbot responde **instantaneamente** de forma estática, **economizando 100% de custos de API (GPT-4o-mini)**.
*   **Reformulação de Consultas Conversacionais**: Quando o colaborador envia perguntas de acompanhamento ou com contexto implícito (ex: "Pego quantos dias?", "Como solicito?"), o sistema utiliza um modelo de linguagem ultrarrápido para expandir e reescrever a pergunta em uma consulta autônoma antes da busca vetorial. Isso garante que o mecanismo de RAG recupere os trechos corretos mesmo em conversas encadeadas, eliminando falsos positivos no guardrail de escopo.
*   **Destacamento de Fatos (Grifo)**: Informações e regras específicas retiradas dos manuais oficiais são destacadas em **negrito** no corpo do texto com índices ordinais de citação (ex: `[1]`, `[2]`).
*   **Painel Retrátil de Auditoria por Seção (Expander)**: Um acordeão retrátil (`st.expander`) intitulado **"Ver fontes da informação"** é renderizado logo abaixo de cada resposta, exibindo o arquivo de origem, a **seção ou subtítulo específico** de onde a regra foi extraída e a **frase exata original** copiada do documento de RH.
*   **Formatação de Mensagens Estruturada (Chat-Native)**: O motor cognitivo utiliza a estrutura nativa de papéis (`SystemMessage`, `HumanMessage`, `AIMessage`) do LangChain. Isso separa estritamente as diretrizes corporativas e regras de formatação (no nível de sistema) dos dados de contexto e histórico conversacional. A LLM utiliza as mensagens anteriores do assistente com metadados como âncoras *few-shot*, eliminando o uso indesejado de emojis e saudações repetitivas de forma extremamente robusta.
*   **Respostas Humanizadas e Empáticas**: A IA é parametrizada com instruções rígidas de compliance (zero alucinação sobre regras) e incentivo à empatia e atendimento amigável aos colaboradores.
*   **Arquitetura Ultra-Leve e Higienizada**: Substituição de banco de dados SQLite complexo e migrações desnecessárias pelo gerenciador de estados em memória nativo do Streamlit (`st.session_state`), reduzindo as linhas de código em mais de 60% e maximizando a velocidade.

---

## 🛠️ Tecnologias Utilizadas (Stack)

*   **Apresentação e Interface**: [Streamlit](https://streamlit.io/) (Interface de chat responsiva e rápida)
*   **Orquestração de RAG e Cadeia de Prompt**: [LangChain](https://www.langchain.com/) (PromptTemplate e abstração de LLM)
*   **Motor Cognitivo (LLM)**: OpenAI GPT-4o-mini via plataforma [OpenRouter](https://openrouter.ai/) (Temperatura de 0.3 para assegurar o equilíbrio entre escrita natural e restrição factual)
*   **Processamento e Embeddings Locais**: HuggingFace `all-MiniLM-L6-v2` (Geração de representações matemáticas de forma local e 100% gratuita)
*   **Banco Vetorial**: [ChromaDB](https://www.trychroma.com/) (Armazenamento de fragmentos de texto locais persistidos em disco)
*   **Gerenciador de Pacotes e Ambiente**: [uv](https://github.com/astral-sh/uv) (Gerenciador ultrarrápido escrito em Rust)

---

## 📁 Estrutura do Projeto

O projeto foi estruturado seguindo o princípio da **Separação de Preocupações (Separation of Concerns)**:

```text
AMCHAN/
├── .streamlit/
│   └── config.toml         # Definições de tema e cores institucionais (Tema Claro)
├── data/
│   └── *.txt               # Documentos originais e manuais de RH (Base de Conhecimento)
├── src/
│   ├── __init__.py
│   ├── database.py         # Lógica de acesso ao ChromaDB, busca com score e embeddings
│   └── llm_service.py      # Configuração do GPT-4o-mini e engenharia de prompt (Guardrails)
├── app.py                  # Ponto de entrada do painel Streamlit e UI do Chat
├── ingest.py               # Script de ETL (Extração, Fragmentação e Vetorização da base)
├── test_run.py             # Script de validação do pipeline RAG via terminal
├── Dockerfile              # Build de imagem Docker com uv
├── .dockerignore
├── .env.example            # Template de credenciais (copiar para .env)
├── .gitignore
├── pyproject.toml          # Definições de dependências estruturadas e linter (Ruff)
├── uv.lock                 # Travamento de dependências pelo uv
└── LICENSE                 # Licença MIT
```

---

## 🚀 Como Executar o Projeto

Você pode rodar esta aplicação localmente utilizando o gerenciador de pacotes **uv** ou de forma rápida em qualquer sistema operacional através do **Docker**.

### Pré-requisitos Gerais
*   Chave de API do **OpenRouter** para acessar a LLM GPT-4o-mini.

---

### Opção A: Executar via Docker (Recomendado para Portabilidade)

O Docker garante que a aplicação rode de forma idêntica em qualquer sistema operacional (Windows, macOS ou Linux), empacotando automaticamente todas as dependências, bibliotecas nativas e a versão correta do Python.

1.  **Construir a Imagem Docker**
    Abra o terminal na raiz do projeto e execute:
    ```bash
    docker build -t assistente-rh-rag .
    ```

2.  **Iniciar o Container**
    Execute o container passando a chave da API do OpenRouter como variável de ambiente e mapeando a porta local `8501`:
    ```bash
    docker run -d -p 8501:8501 -e OPENROUTER_API_KEY="sua_chave_do_openrouter_aqui" --name chatbot-rh assistente-rh-rag
    ```

3.  **Acessar a Aplicação**
    Abra o seu navegador e acesse: [http://localhost:8501](http://localhost:8501).
    *(Nota: Ao iniciar, o container executa automaticamente a ingestão vetorial dos documentos da pasta `data/` antes de subir o painel, garantindo que o assistente já nasça pronto para uso!)*

---

### Opção B: Executar Localmente (Desenvolvimento)

#### Pré-requisitos Locais
*   **Python 3.13** ou superior.
*   **uv** instalado em sua máquina. (Instalação rápida: `curl -LsSf https://astral.sh/uv/install.sh | sh` no macOS/Linux).

#### Passo a Passo

1.  **Instalar Dependências**
    O `uv sync` cria o ambiente virtual automaticamente e instala todas as dependências a partir do lockfile:
    ```bash
    uv sync
    ```

2.  **Configurar Chave da API**
    Copie o template de credenciais e insira sua chave do OpenRouter:
    ```bash
    cp .env.example .env
    ```
    Edite o arquivo `.env` e substitua `sua-chave-aqui` pela sua chave real.

3.  **ETL / Ingestão dos Documentos**
    Popule o banco vetorial local executando o pipeline de ingestão. Ele varrerá o diretório `data/`, fragmentará os textos de RH e criará a base vetorial no diretório `chroma_db/`:
    ```bash
    uv run python ingest.py
    ```

4.  **Iniciar a Aplicação**
    Rode o painel do Streamlit:
    ```bash
    uv run streamlit run app.py
    ```

5.  **Executar Scripts de Teste (Opcional)**
    Você pode testar a cadeia de RAG, a reformulação de consultas e a formatação com citações diretamente pelo terminal:
    ```bash
    uv run python test_run.py
    ```

---

## 🛡️ Guardrails e Segurança (Compliance)

*   **Zero Alucinação**: Se a resposta para a dúvida do colaborador não puder ser atestada com base estrita no contexto vetorial retornado, o assistente responderá polidamente declarando não conter as diretrizes e direcionará o caso para o gestor.
*   **Proteção à Injection**: A engenharia de prompt restringe a IA a atuar puramente como assistente interno, desconsiderando comandos externos do chat que incentivem a mudança de comportamento ou a saída das regras.
*   **Privacidade dos Dados**: O pipeline de embeddings roda em hardware local (CPU) de forma off-line, garantindo que o processamento matemático das informações da empresa não dependa de chamadas ou fluxos para APIs externas de processamento vetorial.

---

## ⚖️ Licença

Este projeto está licenciado sob a licença **MIT** — consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
