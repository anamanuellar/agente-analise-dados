import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
import base64
import re
import streamlit as st

# Configuração de estilo global para gráficos
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("viridis")

# ---------------------------
# Funções de Memória do Agente
# ---------------------------

def add_to_memory(analysis_type, conclusion, data_info=None, plot_data=None):
    """Adiciona conclusão à memória do agente com timestamp e metadados."""
    memory_entry = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': analysis_type,
        'conclusion': conclusion,
        'data_info': data_info or {}
    }
    st.session_state.agent_memory['conclusions'].append(memory_entry)
    st.session_state.agent_memory['analysis_history'].append(analysis_type)
    if plot_data:
        st.session_state.agent_memory['generated_plots'].append(plot_data)


def get_memory_summary():
    """Retorna resumo inteligente da memória do agente."""
    memory = st.session_state.agent_memory
    if not memory['conclusions']:
        return "🤖 **Agente iniciado.** Nenhuma análise realizada ainda."
    summary = f"🧠 **Resumo da Memória do Agente:**\n\n"
    summary += f"- **Total de análises:** {len(memory['conclusions'])}\n"
    summary += f"- **Tipos realizados:** {', '.join(set(memory['analysis_history']))}\n"
    summary += f"- **Última análise:** {memory['conclusions'][-1]['analysis_type']}\n"
    summary += f"- **Período de atividade:** {len(set([c['timestamp'][:10] for c in memory['conclusions']]))} dia(s)\n\n"
    summary += "📊 **Principais Descobertas:**\n"
    for i, conclusion in enumerate(memory['conclusions'][-5:], 1):
        summary += f"{i}. **{conclusion['analysis_type'].title()}:** {conclusion['conclusion']}\n"
    return summary


def get_dataset_info(df):
    """Retorna informações descritivas completas do dataset."""
    info = f"""
📊 **INFORMAÇÕES DESCRITIVAS DO DATASET**

**Dimensões:**
- Linhas: {df.shape[0]:,}
- Colunas: {df.shape[1]}
- Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

**Qualidade dos Dados:**
- Valores nulos: {df.isnull().sum().sum():,}
- Valores únicos (média): {df.nunique().mean():.0f}
- Completude: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%

**Tipos de Colunas:**
- Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}
- Categóricas: {len(df.select_dtypes(include=['object', 'category']).columns)}
- Datetime: {len(df.select_dtypes(include=['datetime']).columns)}

**Estatísticas Gerais:**
- Densidade de dados: {(df.count().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
- Variabilidade média: {df.select_dtypes(include=[np.number]).std().mean():.2f}
"""
    return info


# ---------------------------
# Funções de Análise de Dados
# ---------------------------

def perform_descriptive_analysis(df):
    """Análise descritiva com insights automáticos."""
    cache_key = "descriptive_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]
    desc_stats = df.describe()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    conclusions = []
    for col in numeric_cols[:5]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        median_val = df[col].median()
        if std_val == 0:
            skew_desc = "constante"
        elif abs(mean_val - median_val) > std_val * 0.5:
            skew_desc = "distribuição assimétrica"
        else:
            skew_desc = "distribuição aproximadamente simétrica"
        conclusions.append(f"'{col}': {skew_desc}, média={mean_val:.2f}, mediana={median_val:.2f}")
    conclusion_text = f"Análise descritiva de {len(numeric_cols)} variáveis numéricas. " + "; ".join(conclusions[:3])
    add_to_memory("descriptive_analysis", conclusion_text)
    st.session_state.analysis_cache[cache_key] = desc_stats
    return desc_stats


def analyze_frequent_values(df, max_categories=10):
    """Análise de valores frequentes."""
    cache_key = "frequency_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    discrete_cols = [col for col in df.select_dtypes(include=['int64']).columns if df[col].nunique() <= 20]
    results = {}
    conclusions = []
    for col in categorical_cols:
        if df[col].nunique() <= max_categories:
            value_counts = df[col].value_counts()
            most_freq_val = value_counts.index[0]
            most_freq_count = value_counts.iloc[0]
            conclusions.append(f"Coluna '{col}': valor mais frequente = '{most_freq_val}' ({most_freq_count} ocorrências)")
    if conclusions:
        add_to_memory("frequency_analysis", "; ".join(conclusions))
    st.session_state.analysis_cache[cache_key] = results
    return results


def plot_distribution(df, column):
    """Distribuição e boxplot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(df[column], kde=True, ax=ax1)
    sns.boxplot(y=df[column], ax=ax2)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Mapa de correlação."""
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    plt.tight_layout()
    return fig


def analyze_temporal_patterns(df, time_column):
    """Análise temporal genérica."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[time_column], marker='o', linestyle='', alpha=0.5)
    ax.set_title(f"Padrões Temporais - {time_column}")
    return fig


def perform_clustering_analysis(df):
    """Clustering K-means básico."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "❌ Necessário pelo menos 2 colunas numéricas"
    X = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    conclusion = f"K-means detectou 3 clusters. Tamanhos: {[sum(labels==i) for i in range(3)]}"
    add_to_memory("clustering", conclusion)
    return None, conclusion


def detect_outliers(df, column):
    """Detecção de outliers (IsolationForest)."""
    if column not in df.columns:
        return None, f"Coluna {column} não encontrada"
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(df[[column]].dropna())
    outliers = (preds == -1).sum()
    conclusion = f"Detectados {outliers} outliers na coluna {column}"
    add_to_memory("outlier_detection", conclusion)
    return None, conclusion


def generate_pdf_report(df):
    """Exporta análises para PDF."""
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Relatório de Análise", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def interpret_question(question, df):
    """Interpretação de perguntas."""
    q = question.lower()
    if "correlação" in q:
        return "correlation"
    if "outlier" in q or "anomalia" in q:
        return "outliers"
    if "tempo" in q or "data" in q:
        return "temporal"
    if "cluster" in q:
        return "clustering"
    if "frequente" in q:
        return "frequency"
    if "estatística" in q or "descritiva" in q:
        return "descriptive"
    return "general"


def get_adaptive_suggestions(df):
    """Sugestões automáticas de perguntas."""
    suggestions = []
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        suggestions.append("• *Mostre estatísticas descritivas*")
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        suggestions.append("• *Mostre correlações entre variáveis*")
    if "Class" in df.columns:
        suggestions.append("• *Analise o balanceamento da coluna Class*")
    return suggestions[:6]
