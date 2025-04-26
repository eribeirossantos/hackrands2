import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# Vertex AI para embeddings
from vertexai.language_models import TextEmbeddingModel
import vertexai

# Configuração da API do Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\ranci\\Downloads\\gen-lang-client-0533241835-237a8e1b8397.json"

# Inicializa Vertex AI
@st.cache_resource
def initialize_vertex_ai(project: str, location: str = "us-central1"):
    vertexai.init(project=project, location=location)

# Carregamento dos Dados
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def preprocess_and_vectorize(df, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df['resume_text'])
    job_vector = vectorizer.transform([job_description])
    return vectors, job_vector

def calculate_similarity(vectors, job_vector):
    cosine_similarities = cosine_similarity(vectors, job_vector).flatten()
    return cosine_similarities

def rank_candidates(df, similarities):
    df['similarity'] = similarities
    ranked_df = df.sort_values(by='similarity', ascending=False)
    return ranked_df

def gemini_embedding_similarity(resume_texts, job_description, project: str, location: str = "us-central1"):
    try:
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")

        # Gera embeddings para os currículos
        resume_embeddings = model.get_embeddings(resume_texts)
        resume_vectors = np.array([embedding.values for embedding in resume_embeddings])

        # Embedding da vaga
        job_embedding = model.get_embeddings([job_description])[0].values
        job_vector = np.array(job_embedding)

        # Similaridade cosseno
        similarities = cosine_similarity(resume_vectors, job_vector.reshape(1, -1)).flatten()
        return similarities
    except Exception as e:
        st.error(f"Erro ao gerar embeddings com Vertex AI: {e}")
        return np.zeros(len(resume_texts))

def plot_similarity_bar_chart(df, job_description):
    keywords = set(job_description.lower().split())
    colors = ['#4c72b0' if any(keyword in str(text).lower() for keyword in keywords) else '#dd8452'
              for text in df['resume_text']] # Cores mais contrastantes

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

if uploaded_file:
    df = load_data(uploaded_file)
    st.write(df.head())

    job_description = st.text_area("Descrição da Vaga")
    project = st.text_input("Digite o ID do seu projeto no Google Cloud:")
    location = st.text_input("Localização (ex: us-central1):", value="us-central1")

    if st.button("Classificar Candidatos com TF-IDF"):
        if not job_description:
            st.error("Por favor, preencha a descrição da vaga.")
        else:
            vectors, job_vector = preprocess_and_vectorize(df, job_description)
            similarities = calculate_similarity(vectors, job_vector)
            ranked_candidates = rank_candidates(df, similarities)
            st.write(ranked_candidates[['candidate_name', 'similarity']])
            plot_similarity_bar_chart(ranked_candidates, job_description)

    if st.button("Classificar Candidatos com Gemini Embeddings") and project:
        if not job_description:
            st.error("Por favor, preencha a descrição da vaga.")
        else:
            resume_texts = df['resume_text'].astype(str).tolist()
            initialize_vertex_ai(project, location)
            similarities = gemini_embedding_similarity(resume_texts, job_description, project, location)
            ranked_candidates = rank_candidates(df, similarities)
            st.write(ranked_candidates[['candidate_name', 'similarity']])
            plot_similarity_bar_chart(ranked_candidates, job_description)