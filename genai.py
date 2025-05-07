import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# Google Cloud
from google.oauth2 import service_account
import json

# Vertex AI para embeddings
from vertexai.language_models import TextEmbeddingModel
import vertexai

@st.cache_resource
def get_gcp_credentials() -> Optional[service_account.Credentials]:
    """
    Loads Google Cloud credentials from Streamlit Secrets.
    Returns a credentials object or None if not found or on error.
    """
    if "gcp_service_account" in st.secrets:
        creds_json_str = st.secrets["gcp_service_account"]
        try:
            creds_info = json.loads(creds_json_str)
            credentials = service_account.Credentials.from_service_account_info(creds_info)
            return credentials
        except Exception as e:
            st.error(f"Erro ao carregar credenciais do Streamlit Secrets: {e}. Verifique o formato do JSON.")
            return None
    # Se não estiver no Streamlit Secrets, retorna None, permitindo fallback para ADC (Application Default Credentials)
    return None

# Inicializa Vertex AI
@st.cache_resource
def initialize_vertex_ai(project: str, location: str, credentials: Optional[service_account.Credentials] = None):
    """Initializes the Vertex AI SDK. Raises exception on failure."""
    print(f"Initializing Vertex AI for project '{project}' in location '{location}'...") # Log para debug
    vertexai.init(project=project, location=location, credentials=credentials)
    print("Vertex AI initialized successfully.")

# Carregamento dos Dados
@st.cache_data
def load_data(file: Any) -> Optional[pd.DataFrame]:
    """Loads data from an uploaded CSV file."""
    if file is None:
        return None
    return pd.read_csv(file)

def preprocess_and_vectorize(df: pd.DataFrame, job_description: str) -> Tuple[Any, Any]:
    """Preprocesses resume text and creates TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words='english')
    # Garante que 'resume_text' seja string e lida com NaNs
    resume_texts = df['resume_text'].fillna('').astype(str)
    vectors = vectorizer.fit_transform(resume_texts)
    job_vector = vectorizer.transform([job_description])
    return vectors, job_vector

def calculate_similarity(vectors: Any, job_vector: Any) -> np.ndarray:
    """Calculates cosine similarity between TF-IDF vectors."""
    cosine_similarities = cosine_similarity(vectors, job_vector).flatten()
    return cosine_similarities

def rank_candidates(df: pd.DataFrame, similarities: np.ndarray) -> pd.DataFrame:
    """Adds similarity scores to the DataFrame and ranks candidates."""
    df['similarity'] = similarities
    ranked_df = df.sort_values(by='similarity', ascending=False)
    return ranked_df

def gemini_embedding_similarity(resume_texts: List[str], job_description: str) -> np.ndarray:
    """
    Calculates cosine similarity using Vertex AI embeddings.
    Assumes vertexai.init() has been called successfully beforehand.
    """
    try:
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        batch_size = 250  # Limite da API
        # Limite de caracteres aproximado para evitar erro de token (20k tokens ~ 80k chars, usar 60k por segurança)
        CHAR_LIMIT = 60000 # Reduzido drasticamente para maior segurança contra limite de tokens
        all_resume_embeddings = []
        truncated_indices = [] # Para rastrear quais currículos foram truncados

        # Verifica e trunca a descrição da vaga se necessário
        if len(job_description) > CHAR_LIMIT:
            st.warning(f"Descrição da vaga muito longa ({len(job_description)} caracteres), truncando para {CHAR_LIMIT} caracteres.")
            job_description = job_description[:CHAR_LIMIT]

        # Processa os currículos em lotes
        st.write(f"Processando {len(resume_texts)} currículos em lotes de {batch_size}...") # Feedback para o usuário
        for i in range(0, len(resume_texts), batch_size):
            batch = resume_texts[i:i + batch_size]
            processed_batch = [] # Lote com textos potencialmente truncados
            # Usar um sub-spinner ou apenas logar o progresso
            print(f"Processando lote {i//batch_size + 1}/{(len(resume_texts) + batch_size - 1)//batch_size}...")
            for idx, text in enumerate(batch):
                if len(text) > CHAR_LIMIT:
                    processed_batch.append(text[:CHAR_LIMIT])
                    truncated_indices.append(i + idx) # Guarda o índice original
                else:
                    processed_batch.append(text)
            batch_embeddings = model.get_embeddings(processed_batch) # Usa o lote processado
            all_resume_embeddings.extend(batch_embeddings)
            # Opcional: adicionar um pequeno delay se encontrar erros de rate limit
            # import time
            # time.sleep(1)

        resume_vectors = np.array([embedding.values for embedding in all_resume_embeddings])

        # Embedding da vaga
        job_embedding = model.get_embeddings([job_description])[0].values # A vaga é processada separadamente (1 item)
        job_vector = np.array(job_embedding)

        # Similaridade cosseno
        similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()

        # Informa sobre truncamentos, se houver
        if truncated_indices:
            st.warning(f"{len(set(truncated_indices))} currículo(s) excederam {CHAR_LIMIT} caracteres e foram truncados antes do embedding. Isso pode afetar a pontuação de similaridade.")

        return similarities
    except Exception as e:
        st.error(f"Erro ao gerar embeddings com Vertex AI: {e}")
        return np.zeros(len(resume_texts))

def plot_similarity_bar_chart(df: pd.DataFrame, job_description: str):
    """Plots a horizontal bar chart of candidate similarities."""
    keywords = set(job_description.lower().split())
    # Garante que 'resume_text' seja string e lida com NaNs antes de verificar keywords
    colors = ['#4c72b0' if any(keyword in str(text).lower() for keyword in keywords) else '#dd8452'
              for text in df['resume_text'].fillna('')] # Cores mais contrastantes

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5))) # Ajuste um pouco maior para os rótulos
    bars = ax.barh(df['candidate_name'], df['similarity'], color=colors)

    # Rótulos nos eixos
    ax.set_xlabel("Similaridade", fontsize=12)
    ax.set_ylabel("Candidato", fontsize=12)
    ax.set_title("Similaridade dos Candidatos com a Vaga", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Adicionar rótulos nas barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', ha='left', fontsize=10, color='black')

    # Ajustar limites do eixo x para acomodar os rótulos
    max_similarity = df['similarity'].max()
    ax.set_xlim(0, max(0.5, max_similarity + 0.1))

    # Remover spines desnecessárias para um visual mais limpo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)

# Interface do Streamlit
st.title("Automatização de Triagem de Candidatos")

uploaded_file = st.file_uploader("Carregar currículos (CSV)", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Verifica se o DataFrame foi carregado e tem as colunas necessárias
    if df is None:
        st.warning("Falha ao carregar o arquivo CSV.")
    elif not all(col in df.columns for col in ['candidate_name', 'resume_text']):
        st.error("O arquivo CSV deve conter as colunas 'candidate_name' e 'resume_text'.")
        st.stop() # Impede a execução do restante se as colunas estiverem faltando
    else:
        st.write(f"Pré-visualização dos Dados Carregados: ({len(df)} registros encontrados)") # Adiciona a contagem total
        st.dataframe(df.head())

        job_description = st.text_area("Descrição da Vaga", height=200)

        # --- Opção TF-IDF ---
        st.markdown("---")
        st.markdown("### Classificação com TF-IDF")
        if st.button("Classificar com TF-IDF"):
            if not job_description:
                st.error("Por favor, preencha a descrição da vaga.")
            else:
                with st.spinner("Calculando similaridade TF-IDF..."):
                    vectors, job_vector = preprocess_and_vectorize(df, job_description)
                    similarities_tfidf = calculate_similarity(vectors, job_vector)
                    # Usar cópia para não alterar df original em cache
                    ranked_candidates_tfidf = rank_candidates(df.copy(), similarities_tfidf)
                    st.write("Resultados TF-IDF:")
                    # Limitar a exibição e plotagem para datasets grandes
                    display_limit = 100 # Reduzir limite para plotagem ser mais viável
                    st.dataframe(ranked_candidates_tfidf[['candidate_name', 'similarity']].head(display_limit))
                    if len(ranked_candidates_tfidf) <= display_limit:
                        plot_similarity_bar_chart(ranked_candidates_tfidf, job_description)
                    else:
                        st.info(f"Mostrando os top {display_limit} resultados na tabela e no gráfico devido ao grande número total de candidatos ({len(ranked_candidates_tfidf)}).")
                        plot_similarity_bar_chart(ranked_candidates_tfidf.head(display_limit), job_description) # Plotar apenas o head

        # --- Opção Gemini Embeddings ---
        st.markdown("---")
        st.markdown("### Classificação com Gemini Embeddings (Vertex AI)")
        project_id = st.text_input("ID do Projeto Google Cloud:")
        location = st.text_input("Região (ex: us-central1):", value="us-central1")

        if st.button("Classificar com Gemini Embeddings"):
            if not job_description:
                st.error("Por favor, preencha a descrição da vaga.")
            elif not project_id:
                st.error("Por favor, digite o ID do Projeto Google Cloud.")
            else:
                gcp_creds = None
                if "gcp_service_account" in st.secrets:
                    gcp_creds = get_gcp_credentials()
                    if gcp_creds is None:
                        # Erro já foi exibido por get_gcp_credentials, então paramos.
                        st.stop()
                else:
                    st.info(
                        "Nenhuma credencial 'gcp_service_account' encontrada no Streamlit Secrets. "
                        "Vertex AI tentará usar credenciais padrão do ambiente (ADC), se disponíveis (útil para desenvolvimento local)."
                    )

                with st.spinner("Inicializando Vertex AI e calculando embeddings..."):
                    try:
                        initialize_vertex_ai(project=project_id, location=location, credentials=gcp_creds)
                        
                        resume_texts = df['resume_text'].fillna('').astype(str).tolist()
                        similarities_gemini = gemini_embedding_similarity(resume_texts, job_description)
                        ranked_candidates_gemini = rank_candidates(df.copy(), similarities_gemini)
                        
                        st.write("Resultados Gemini Embeddings:")
                        display_limit = 100 
                        st.dataframe(ranked_candidates_gemini[['candidate_name', 'similarity']].head(display_limit))
                        if len(ranked_candidates_gemini) <= display_limit:
                            plot_similarity_bar_chart(ranked_candidates_gemini, job_description)
                        else:
                            st.info(f"Mostrando os top {display_limit} resultados na tabela e no gráfico devido ao grande número total de candidatos ({len(ranked_candidates_gemini)}).")
                            plot_similarity_bar_chart(ranked_candidates_gemini.head(display_limit), job_description)
                    except Exception as e:
                        st.error(f"Falha na classificação com Gemini: {e}")
                        st.warning("Verifique o ID do projeto, região. Se estiver usando Streamlit Secrets, certifique-se de que 'gcp_service_account' está configurado corretamente e que a conta de serviço tem as permissões necessárias (ex: 'Vertex AI User') e o faturamento está ativo no projeto.")
else:
    st.info("Por favor, carregue um arquivo CSV com os currículos.")

# Textos Explicativos
st.markdown("---")
# Usar st.expander para ocultar o texto por padrão
with st.expander("Entenda os Métodos de Classificação Utilizados"):
    st.markdown("""
    **Classificação com TF-IDF (Term Frequency-Inverse Document Frequency):**
    *   Este método clássico de recuperação de informação avalia a importância de uma palavra em um documento (currículo) em relação a uma coleção de documentos.
    *   **Prós:** Rápido, eficiente e ótimo para encontrar correspondências exatas de palavras-chave entre o currículo e a descrição da vaga. Não requer APIs externas.
    *   **Contras:** Não compreende o significado semântico ou o contexto. Sinônimos (ex: "engenheiro de software" vs. "desenvolvedor") ou conceitos relacionados podem não ser bem capturados se as palavras exatas não estiverem presentes.

    **Classificação com Gemini Embeddings (Vertex AI):**
    *   Utiliza modelos de linguagem avançados (como o Gemini via Vertex AI) para converter texto (currículos e descrição da vaga) em vetores numéricos (embeddings) que representam seu significado semântico. A similaridade é calculada com base na proximidade desses vetores no espaço.
    *   **Prós:** Excelente em capturar o significado, contexto e nuances da linguagem. Consegue identificar candidatos relevantes mesmo que não usem as mesmas palavras-chave da descrição da vaga, entendendo sinônimos e conceitos relacionados.
    *   **Contras:** Requer configuração e acesso à API do Google Cloud (Vertex AI), incluindo um projeto com faturamento ativo. Pode ser um pouco mais lento que o TF-IDF devido à chamada de API externa.
    """)