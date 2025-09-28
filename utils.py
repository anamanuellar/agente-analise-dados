# utils.py - Arquivo completo com todas as funções necessárias
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

class HybridGeminiAgent:
    """
    Agente Híbrido que combina LLM (Gemini) com funções robustas de análise
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.model = None
        self.conversation_history = []
        self.dataset_context = {}
        self.api_key = None
        
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4000,
        }
        
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    
    def configure_gemini(self, api_key=None):
        """Configura o Gemini com a API Key"""
        try:
            if api_key:
                self.api_key = api_key
            elif hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                self.api_key = st.secrets['GEMINI_API_KEY']
            elif hasattr(st, 'secrets') and 'gemini' in st.secrets and 'api_key' in st.secrets['gemini']:
                self.api_key = st.secrets['gemini']['api_key']
            else:
                return False
            
            genai.configure(api_key=self.api_key)
            
            self.model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            
            # Teste rápido
            test_response = self.model.generate_content("OK")
            return True
                
        except Exception as e:
            st.error(f"Erro ao configurar Gemini: {str(e)}")
            self.model = None
            return False
    
    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        """Chama Gemini de forma segura"""
        if not self.model:
            if not self.configure_gemini():
                return "Gemini não configurado"
        
        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(full_prompt)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "Resposta vazia do Gemini"
                
        except Exception as e:
            return f"Erro na LLM: {str(e)}"
    
    def check_configuration(self):
        """Verifica se o Gemini está configurado"""
        if not self.model:
            return False, "Modelo não inicializado"
        
        try:
            test_response = self.model.generate_content("Teste")
            return True, "Configuração OK"
        except Exception as e:
            return False, f"Erro: {str(e)}"
    
    def analyze_dataset_initially(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise inicial do dataset"""
        basic_analysis = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_cols": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum(),
            "completeness": ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100),
            "duplicates": df.duplicated().sum()
        }
        
        system_context = "Você é um especialista em análise de dados."
        
        prompt = f"""
        Analise este dataset:
        Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas
        Colunas: {list(df.columns)[:5]}
        Tipos: {basic_analysis['numeric_cols']} numéricas, {basic_analysis['categorical_cols']} categóricas
        
        Forneça:
        1. Tipo de dataset
        2. 3 características principais
        3. 3 análises recomendadas
        """
        
        llm_response = self._call_gemini(prompt, system_context)
        
        result = {
            **basic_analysis,
            "llm_analysis": llm_response,
            "dataset_type": self._extract_dataset_type(llm_response),
            "key_characteristics": ["Análise disponível"],
            "recommended_analyses": ["Estatísticas descritivas", "Correlações"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.dataset_context = result
        return result
    
    def interpret_query_intelligently(self, user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Interpreta query do usuário"""
        system_context = "Classifique esta pergunta sobre análise de dados."
        
        prompt = f"""
        Pergunta: "{user_query}"
        Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas
        
        Classifique em UMA categoria:
        - descriptive: estatísticas básicas
        - correlation: correlações
        - distribution: distribuições
        - outliers: outliers
        - clustering: agrupamentos
        - temporal: padrões temporais
        - frequency: frequências
        - balance: balanceamento
        - insights: conclusões
        - memory: memória
        
        Responda apenas: categoria|coluna (se aplicável)
        """
        
        llm_classification = self._call_gemini(prompt, system_context)
        
        try:
            if '|' in llm_classification:
                category, columns_str = llm_classification.split('|', 1)
                specific_columns = [col.strip() for col in columns_str.split(',') if col.strip()]
            else:
                category = llm_classification.strip()
                specific_columns = []
        except:
            category = self._fallback_classification(user_query.lower())
            specific_columns = []
        
        return {
            "category": category,
            "specific_columns": specific_columns,
            "original_query": user_query,
            "llm_interpretation": llm_classification,
            "confidence": "high" if '|' in llm_classification else "medium"
        }
    
    def generate_intelligent_response(self, query_result: Dict, analysis_result: Any, df: pd.DataFrame) -> str:
        """Gera resposta inteligente"""
        system_context = "Você é um consultor de dados. Transforme resultados técnicos em insights claros."
        
        prompt = f"""
        Pergunta: "{query_result['original_query']}"
        Tipo: {query_result['category']}
        Resultado: {str(analysis_result)[:500] if analysis_result else "Análise executada"}
        
        Gere resposta clara:
        1. Responda à pergunta
        2. Principais achados
        3. Insights práticos
        
        Máximo 200 palavras.
        """
        
        return self._call_gemini(prompt, system_context)
    
    def generate_smart_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Gera sugestões inteligentes"""
        prompt = f"""
        Dataset: {df.shape}
        Colunas: {list(df.columns)[:5]}
        
        Gere 5 perguntas práticas para análise.
        Formato: uma por linha com "•"
        """
        
        response = self._call_gemini(prompt)
        
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                suggestion = line[1:].strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:5] if suggestions else [
            "Mostre estatísticas descritivas",
            "Analise correlações",
            "Detecte outliers",
            "Faça clustering",
            "Mostre frequências"
        ]
    
    def generate_executive_conclusions(self, df: pd.DataFrame, memory: Dict) -> str:
        """Gera conclusões executivas"""
        prompt = f"""
        Dataset: {df.shape[0]:,} registros, {df.shape[1]} variáveis
        Análises realizadas: {len(memory.get('conclusions', []))}
        
        Gere conclusões executivas:
        ## Resumo Executivo
        ## Principais Descobertas
        ## Recomendações
        """
        
        return self._call_gemini(prompt)
    
    def _extract_dataset_type(self, text: str) -> str:
        """Extrai tipo de dataset"""
        text_lower = text.lower()
        if 'fraude' in text_lower:
            return 'Detecção de Fraude'
        elif 'vendas' in text_lower or 'sales' in text_lower:
            return 'Dados de Vendas'
        elif 'financeiro' in text_lower:
            return 'Dados Financeiros'
        return 'Dataset Genérico'
    
    def _fallback_classification(self, query_lower: str) -> str:
        """Classificação de fallback"""
        if any(word in query_lower for word in ['correlação', 'relaciona']):
            return 'correlation'
        elif any(word in query_lower for word in ['outlier', 'anomalia']):
            return 'outliers'
        elif any(word in query_lower for word in ['cluster', 'agrupamento']):
            return 'clustering'
        elif any(word in query_lower for word in ['distribuição', 'histograma']):
            return 'distribution'
        elif any(word in query_lower for word in ['frequente', 'comum']):
            return 'frequency'
        elif any(word in query_lower for word in ['tempo', 'temporal']):
            return 'temporal'
        elif any(word in query_lower for word in ['balanceamento']):
            return 'balance'
        elif any(word in query_lower for word in ['conclusão', 'insight']):
            return 'insights'
        elif any(word in query_lower for word in ['memória', 'histórico']):
            return 'memory'
        else:
            return 'descriptive'

# FUNÇÃO PARA INICIALIZAR O AGENTE
def initialize_hybrid_agent():
    """Inicializa o agente híbrido"""
    if 'hybrid_agent' not in st.session_state:
        st.session_state.hybrid_agent = HybridGeminiAgent()
    
    agent = st.session_state.hybrid_agent
    is_configured, status = agent.check_configuration()
    
    if not is_configured:
        success = agent.configure_gemini()
        if not success:
            # Não bloquear - apenas avisar
            pass
    
    return agent

# VERIFICAÇÃO DE SECRETS
def verificar_secrets():
    """Verifica se os secrets estão configurados"""
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            key = st.secrets['GEMINI_API_KEY']
            if key and len(key) > 30:
                return True, "GEMINI_API_KEY encontrada"
            else:
                return False, "GEMINI_API_KEY muito curta"
        
        return False, "Nenhuma API key encontrada"
        
    except Exception as e:
        return False, f"Erro ao verificar secrets: {str(e)}"

# FUNÇÕES DE MEMÓRIA
def add_to_memory(analysis_type, conclusion, data_info=None, plot_data=None):
    """Adiciona conclusão à memória"""
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
    """Retorna resumo da memória"""
    memory = st.session_state.agent_memory
    if not memory['conclusions']:
        return "Agente iniciado. Nenhuma análise realizada ainda."
    
    summary = f"**Resumo da Memória:**\n\n"
    summary += f"- **Total de análises:** {len(memory['conclusions'])}\n"
    summary += f"- **Tipos realizados:** {', '.join(set(memory['analysis_history']))}\n"
    summary += f"- **Última análise:** {memory['conclusions'][-1]['analysis_type']}\n"
    
    summary += "\n**Principais Descobertas:**\n"
    for i, conclusion in enumerate(memory['conclusions'][-3:], 1):
        summary += f"{i}. **{conclusion['analysis_type'].title()}:** {conclusion['conclusion'][:80]}...\n"
    
    return summary

# FUNÇÕES DE ANÁLISE
def get_dataset_info(df):
    """Informações descritivas do dataset"""
    info = f"""
**INFORMAÇÕES DO DATASET**

**Dimensões:**
- Linhas: {df.shape[0]:,}
- Colunas: {df.shape[1]}
- Tamanho em memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

**Qualidade dos Dados:**
- Valores nulos: {df.isnull().sum().sum():,}
- Completude: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%

**Tipos de Colunas:**
- Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}
- Categóricas: {len(df.select_dtypes(include=['object', 'category']).columns)}
"""
    return info

def perform_descriptive_analysis(df):
    """Análise descritiva robusta"""
    cache_key = "descriptive_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    desc_stats = df.describe()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    conclusions = []
    
    for col in numeric_cols[:3]:
        mean_val = df[col].mean()
        median_val = df[col].median()
        conclusions.append(f"'{col}': média={mean_val:.2f}, mediana={median_val:.2f}")
    
    conclusion_text = f"Análise descritiva de {len(numeric_cols)} variáveis. " + "; ".join(conclusions)
    add_to_memory("descriptive_analysis", conclusion_text, {
        "numeric_columns": len(numeric_cols),
        "total_rows": len(df)
    })
    
    st.session_state.analysis_cache[cache_key] = desc_stats
    return desc_stats

def plot_correlation_heatmap(df):
    """Gera mapa de correlação"""
    cache_key = "correlation_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None, "Necessário pelo menos 2 colunas numéricas"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', 
                center=0, square=True, fmt='.2f', ax=ax)
    
    ax.set_title('Matriz de Correlação')
    plt.tight_layout()
    
    # Análise das correlações
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    conclusion = f"Análise de correlação entre {len(numeric_cols)} variáveis: {len(high_corr_pairs)} correlações altas (>0.7)"
    
    add_to_memory("correlation_analysis", conclusion)
    
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def plot_distribution(df, column):
    """Gera histograma"""
    cache_key = f"distribution_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histograma
    sns.histplot(df[column].dropna(), kde=True, ax=ax1)
    ax1.set_title(f'Distribuição de {column}')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=df[column].dropna(), ax=ax2)
    ax2.set_title(f'Box Plot - {column}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Estatísticas
    data_clean = df[column].dropna()
    skewness = data_clean.skew()
    
    conclusion = f"Distribuição de '{column}': skewness={skewness:.2f}"
    
    add_to_memory("distribution_analysis", conclusion, {"column": column})
    
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def detect_outliers(df, column):
    """Detecção de outliers"""
    cache_key = f"outlier_detection_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"Coluna '{column}' não encontrada"
    
    data = df[[column]].dropna()
    
    if len(data) < 10:
        return None, f"Poucos dados na coluna '{column}'"

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso_forest.fit_predict(data)
    
    # IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
    
    outlier_count_iso = sum(1 for x in outliers_iso if x == -1)
    outlier_count_iqr = sum(outliers_iqr)
    
    conclusion = f"Outliers em '{column}': Isolation Forest: {outlier_count_iso}, IQR: {outlier_count_iqr}"
    
    add_to_memory("outlier_detection", conclusion, {"column": column})
    
    st.session_state.analysis_cache[cache_key] = (None, conclusion)
    return None, conclusion

def perform_clustering_analysis(df, n_clusters=None, sample_size=5000):
    """Análise de clustering"""
    cache_key = "clustering_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "Necessário pelo menos 2 colunas numéricas"
    
    # Amostragem
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    df_numeric = df_sample[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric.fillna(df_numeric.mean()))
    
    if n_clusters is None:
        silhouette_scores = []
        K_range = range(2, min(8, len(df_sample)//50 + 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        n_clusters = K_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        best_silhouette = silhouette_score(X_scaled, cluster_labels)
    
    conclusion = f"Clustering identificou {n_clusters} grupos com silhouette score: {best_silhouette:.3f}"
    
    add_to_memory("clustering_analysis", conclusion)
    
    st.session_state.analysis_cache[cache_key] = (None, conclusion)
    return None, conclusion

def analyze_frequent_values(df, max_categories=10):
    """Análise de frequência"""
    cache_key = "frequency_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    results = {}
    for col in categorical_cols:
        if df[col].nunique() <= max_categories:
            value_counts = df[col].value_counts()
            results[col] = {
                'type': 'categorical',
                'most_frequent': value_counts.head(3).to_dict(),
                'least_frequent': value_counts.tail(3).to_dict(),
                'unique_count': df[col].nunique(),
                'null_count': df[col].isnull().sum()
            }
    
    if results:
        conclusion_text = f"Análise de frequência: {len(results)} colunas analisadas"
        add_to_memory("frequency_analysis", conclusion_text)
    
    st.session_state.analysis_cache[cache_key] = results
    return results

def analyze_balance(df, column):
    """Análise de balanceamento"""
    cache_key = f"balance_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"Coluna '{column}' não encontrada"
    
    unique_values = df[column].nunique()
    if unique_values != 2:
        return None, f"Coluna '{column}' não é binária"
    
    value_counts = df[column].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Balanceamento da Coluna {column}')
    plt.tight_layout()
    
    conclusion = f"Balanceamento da coluna '{column}': {value_counts.to_dict()}"
    
    add_to_memory("balance_analysis", conclusion, {"column": column})
    
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def analyze_temporal_patterns(df, time_column):
    """Análise temporal"""
    cache_key = f"temporal_analysis_{time_column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if time_column not in df.columns:
        return None, f"Coluna '{time_column}' não encontrada"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histograma temporal
    ax1.hist(df[time_column].dropna(), bins=50, alpha=0.7)
    ax1.set_title(f'Distribuição Temporal - {time_column}')
    ax1.grid(True, alpha=0.3)
    
    # Padrões por classe se existir
    if 'Class' in df.columns:
        for class_val in sorted(df['Class'].unique()):
            subset = df[df['Class'] == class_val]
            ax2.hist(subset[time_column].dropna(), bins=30, alpha=0.6, 
                    label=f'Classe {class_val}')
        ax2.legend()
    else:
        ax2.plot(df[time_column].dropna())
    
    ax2.set_title('Padrões Temporais')
    ax2.grid(True, alpha=0.3)
    
    # Box plot
    ax3.boxplot(df[time_column].dropna())
    ax3.set_title(f'Box Plot - {time_column}')
    ax3.grid(True, alpha=0.3)
    
    # Linha temporal
    time_sorted = df.sort_values(time_column)
    ax4.plot(time_sorted[time_column].values)
    ax4.set_title('Evolução Temporal')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    conclusion = f"Análise temporal de '{time_column}' com {len(df[time_column].dropna())} registros"

    add_to_memory("temporal_analysis", conclusion, {"time_column": time_column})

    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def generate_complete_analysis_summary(df):
    """Gera resumo completo"""
    summary_text = f"""
**RESUMO COMPLETO DAS ANÁLISES**

{get_dataset_info(df)}

**ANÁLISES REALIZADAS:**
"""
    
    for i, conclusion_entry in enumerate(st.session_state.agent_memory['conclusions'], 1):
        analysis_name = conclusion_entry['analysis_type'].replace('_', ' ').title()
        summary_text += f"\n**{i}. {analysis_name}:**\n"
        summary_text += f"   {conclusion_entry['conclusion']}\n"
    
    summary_text += f"""

**TOTAL DE ANÁLISES:** {len(st.session_state.agent_memory['conclusions'])}
**GERADO EM:** {datetime.now().strftime("%d/%m/%Y às %H:%M:%S")}
"""
    return summary_text

def generate_pdf_report(df, gemini_insights=None):
    """Gera PDF"""
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Página de capa
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.8, 'RELATÓRIO AGENTE HÍBRIDO', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.7, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                ha='center', va='center', fontsize=12)
        
        dataset_info = get_dataset_info(df)
        ax.text(0.1, 0.5, dataset_info, ha='left', va='top', fontsize=10)
        
        if gemini_insights:
            ax.text(0.1, 0.3, f"Insights da IA:\n{gemini_insights[:300]}...", 
                   ha='left', va='top', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        # Página de resumo
        analysis_summary = generate_complete_analysis_summary(df)
        
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.text(0.05, 0.95, 'RESUMO DAS ANÁLISES', 
               ha='left', va='top', fontsize=14, fontweight='bold')
        ax.text(0.05, 0.85, analysis_summary[:2000], ha='left', va='top', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# FUNÇÕES DE COMPATIBILIDADE
def interpret_question(question, df):
    """Função de compatibilidade - classificação simples"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['correlação', 'relaciona']):
        return 'correlation'
    elif any(word in question_lower for word in ['outlier', 'anomalia']):
        return 'outliers'
    elif any(word in question_lower for word in ['cluster', 'agrupamento']):
        return 'clustering'
    elif any(word in question_lower for word in ['distribuição', 'histograma']):
        return 'distribution'
    elif any(word in question_lower for word in ['frequente', 'comum']):
        return 'frequency'
    elif any(word in question_lower for word in ['tempo', 'temporal']):
        return 'temporal'
    elif any(word in question_lower for word in ['balanceamento']):
        return 'balance'
    elif any(word in question_lower for word in ['conclusão', 'insight']):
        return 'insights'
    elif any(word in question_lower for word in ['memória', 'histórico']):
        return 'memory'
    else:
        return 'descriptive'

def get_adaptive_suggestions(df):
    """Sugestões adaptativas básicas"""
    suggestions = [
        "• Mostre estatísticas descritivas",
        "• Mostre correlações entre variáveis", 
        "• Faça clustering automático dos dados",
        "• Detecte outliers nas colunas numéricas",
        "• Mostre valores mais frequentes",
        "• Qual sua memória de análises?"
    ]
    
    # Adicionar sugestões específicas baseadas nas colunas
    if 'Time' in df.columns:
        suggestions.append("• Analise padrões temporais na coluna Time")
    
    if 'Amount' in df.columns:
        suggestions.append("• Qual a distribuição da coluna Amount?")
    
    if 'Class' in df.columns:
        suggestions.append("• Analise o balanceamento da coluna Class")
    
    return suggestions[:6]
