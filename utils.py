import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import json
from datetime import datetime
import base64
import re
import streamlit as st
import google.generativeai as genai
from typing import Dict, List, Any, Tuple
from matplotlib.backends.backend_pdf import PdfPages

# ========================================================================
# CLASSE PRINCIPAL: GEMINI AGENT (CÉREBRO LLM)
# ========================================================================

class GeminiAgent:
    """Agente que usa Google Gemini como cérebro do sistema"""

    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.conversation_history = []
        self.dataset_context = {}

        # Configurações de segurança
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        # Configurações de geração
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4000,
        }

    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        """Chama Gemini"""
        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return response.text
        except Exception as e:
            st.error(f"Erro no Gemini: {e}")
            return f"[Erro Gemini] {e}"

    def analyze_dataset_initially(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise inicial com Gemini"""
        system_context = """Você é um especialista em análise de dados. Analise datasets CSV e forneça insights profissionais e práticos."""
        prompt = f"""
        Analise este dataset:
        - Linhas: {df.shape[0]:,}
        - Colunas: {df.shape[1]}
        Colunas: {list(df.columns)}
        Estatísticas: {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "Sem colunas numéricas"}
        """
        response = self._call_gemini(prompt, system_context)
        self.dataset_context = {"initial_analysis": response}
        return {"initial_analysis": response}

    def process_user_query(self, user_query: str, df: pd.DataFrame, memory: Dict) -> Tuple[str, Dict]:
        """Processa pergunta do usuário, combinando Gemini + funções locais"""
        self.conversation_history.append({"role": "user", "content": user_query})

        # Decidir análise (simples, pode ser refinado com LLM também)
        analysis_type = interpret_question(user_query, df)

        # Executar função local se aplicável
        fig, conclusion = None, None
        if analysis_type == "descriptive":
            perform_descriptive_analysis(df)
            conclusion = get_dataset_info(df)
        elif analysis_type == "correlation":
            fig, conclusion = plot_correlation_heatmap(df)
        elif analysis_type == "clustering":
            _, conclusion = perform_clustering_analysis(df)
        elif analysis_type == "outliers":
            col = df.select_dtypes(include=[np.number]).columns[0]
            _, conclusion = detect_outliers(df, col)
        elif analysis_type == "distribution":
            col = df.select_dtypes(include=[np.number]).columns[0]
            fig, conclusion = plot_distribution(df, col)
        elif analysis_type == "balance":
            col = [c for c in df.columns if df[c].nunique() == 2][0]
            fig, conclusion = analyze_balance(df, col)
        else:
            conclusion = "Análise geral executada."

        # Resposta final com Gemini
        system_context = """Você é um consultor de dados. Explique resultados técnicos de forma clara e mostre insights de negócio."""
        prompt = f"""
        Pergunta: {user_query}
        Tipo de análise: {analysis_type}
        Resultado técnico: {conclusion}
        """
        final_response = self._call_gemini(prompt, system_context)

        self.conversation_history.append({"role": "assistant", "content": final_response})
        return final_response, {"figure": fig}

# ========================================================================
# FUNÇÕES UTILITÁRIAS ORIGINAIS (estatísticas, gráficos, clustering, etc.)
# ========================================================================

def add_to_memory(analysis_type, conclusion, data_info=None, plot_data=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": analysis_type,
        "conclusion": conclusion,
        "data_info": data_info or {},
    }
    st.session_state.agent_memory["conclusions"].append(entry)
    st.session_state.agent_memory["analysis_history"].append(analysis_type)
    if plot_data:
        st.session_state.agent_memory["generated_plots"].append(plot_data)

def get_dataset_info(df):
    return f"""
📊 Dataset: {df.shape[0]:,} linhas × {df.shape[1]} colunas
- Nulos: {df.isnull().sum().sum()}
- Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}
- Categóricas: {len(df.select_dtypes(include=['object']).columns)}
"""

def perform_descriptive_analysis(df):
    desc_stats = df.describe()
    add_to_memory("descriptive_analysis", f"Análise de {len(df.columns)} variáveis.", {"rows": len(df)})
    return desc_stats

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "Poucas colunas numéricas"
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax)
    conclusion = f"Matriz de correlação com {len(numeric_cols)} variáveis."
    add_to_memory("correlation_analysis", conclusion)
    return fig, conclusion

def perform_clustering_analysis(df, n_clusters=2):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "Poucas colunas numéricas"
    X = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    conclusion = f"KMeans com {n_clusters} clusters. Silhouette={silhouette:.3f}"
    add_to_memory("clustering_analysis", conclusion)
    return None, conclusion

def detect_outliers(df, column):
    data = df[column].dropna()
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(data.values.reshape(-1, 1))
    outliers = (preds == -1).sum()
    conclusion = f"{outliers} outliers detectados em {column}."
    add_to_memory("outlier_detection", conclusion)
    return None, conclusion

def plot_distribution(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    conclusion = f"Distribuição da coluna {column}."
    add_to_memory("distribution_analysis", conclusion)
    return fig, conclusion

def analyze_balance(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    conclusion = f"Balanceamento da coluna {column}."
    add_to_memory("balance_analysis", conclusion)
    return fig, conclusion

def generate_pdf_report(df, llm_insights=None):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.9, "Relatório de Análises", ha="center", fontsize=14, fontweight="bold")
        ax.text(0.05, 0.8, get_dataset_info(df), fontsize=10, va="top")
        if llm_insights:
            ax.text(0.05, 0.6, "Insights da LLM:", fontsize=12, fontweight="bold")
            ax.text(0.05, 0.55, llm_insights, fontsize=9, va="top")
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ========================================================================
# INTERPRETAÇÃO DE PERGUNTAS (MAPEAMENTO SIMPLES)
# ========================================================================

def interpret_question(question, df):
    q = question.lower()
    if "correlação" in q: return "correlation"
    if "cluster" in q: return "clustering"
    if "outlier" in q: return "outliers"
    if "distribuição" in q: return "distribution"
    if "balance" in q: return "balance"
    if "tempo" in q: return "temporal"
    return "descriptive"
