import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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
        vectors, job_vector = preprocess_and_vectorize(df, job_description)
        similarities = calculate_similarity(vectors, job_vector)
        ranked_candidates = rank_candidates(df, similarities)
        st.write(ranked_candidates[['candidate_name', 'similarity']])

    if st.button("Classificar Candidatos com Gemini Embeddings") and project:
        resume_texts = df['resume_text'].astype(str).tolist()
        initialize_vertex_ai(project, location)
        similarities = gemini_embedding_similarity(resume_texts, job_description, project, location)
        ranked_candidates = rank_candidates(df, similarities)
        st.write(ranked_candidates[['candidate_name', 'similarity']])
