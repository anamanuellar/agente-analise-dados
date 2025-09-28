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

# === ARQUITETURA HÍBRIDA ===

class HybridGeminiAgent:
    """
    Agente Híbrido que combina:
    - LLM (Gemini) para interpretação e insights
    - Funções robustas para execução das análises
    - Interface completa da versão 1
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.model = None
        self.conversation_history = []
        self.dataset_context = {}
        
        # Configurações do Gemini
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2000,
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
    
    def configure_gemini(self, api_key: str):
        """Configura o Gemini com a API Key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config
        )
    
    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        """Chama Gemini de forma segura com fallback"""
        if not self.model:
            return "Gemini não configurado"
            
        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            st.error(f"Erro no Gemini: {e}")
            return f"Erro na LLM: {str(e)}"
    
    def analyze_dataset_initially(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise inicial com Gemini + dados estruturados robustos"""
        
        # PARTE 1: Análise robusta (sempre funciona)
        basic_analysis = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_cols": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum(),
            "completeness": ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100),
            "duplicates": df.duplicated().sum()
        }
        
        # PARTE 2: Inteligência com Gemini (com fallback)
        system_context = """Você é um especialista em análise de dados. 
        Analise o dataset e forneça insights contextuais em português."""
        
        prompt = f"""
        Analise este dataset CSV:
        
        Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas
        Colunas: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}
        Tipos: {basic_analysis['numeric_cols']} numéricas, {basic_analysis['categorical_cols']} categóricas
        Qualidade: {basic_analysis['completeness']:.1f}% completo, {basic_analysis['duplicates']} duplicatas
        
        Amostra:
        {df.head(3).to_string()}
        
        Forneça uma análise estruturada:
        1. TIPO DE DATASET: Identifique o domínio (fraude, vendas, etc.)
        2. CARACTERÍSTICAS PRINCIPAIS: 3 pontos importantes
        3. ANÁLISES RECOMENDADAS: 3 análises mais valiosas
        4. INSIGHTS POTENCIAIS: O que pode ser descoberto
        
        Seja conciso e focado em valor prático.
        """
        
        llm_response = self._call_gemini(prompt, system_context)
        
        # Combinar análises
        result = {
            **basic_analysis,
            "llm_analysis": llm_response,
            "dataset_type": self._extract_dataset_type(llm_response),
            "key_characteristics": self._extract_characteristics(llm_response),
            "recommended_analyses": self._extract_recommendations(llm_response),
            "timestamp": datetime.now().isoformat()
        }
        
        self.dataset_context = result
        return result
    
    def interpret_query_intelligently(self, user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        HÍBRIDO: LLM interpreta + mapeamento para funções robustas
        """
        
        # PARTE 1: Interpretação inteligente com Gemini
        system_context = """Você é um especialista em interpretação de queries sobre análise de dados.
        Classifique a pergunta e forneça contexto."""
        
        prompt = f"""
        Pergunta do usuário: "{user_query}"
        
        Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas
        Colunas disponíveis: {list(df.columns)[:10]}
        
        Classifique a pergunta em UMA categoria:
        - descriptive: estatísticas básicas, resumos
        - correlation: correlações, relacionamentos
        - distribution: distribuições, histogramas
        - outliers: outliers, anomalias
        - clustering: agrupamentos, segmentação
        - temporal: padrões temporais, trends
        - frequency: valores frequentes, contagens
        - balance: balanceamento de classes
        - insights: conclusões, descobertas
        - memory: histórico, memória do agente
        
        Responda APENAS com a categoria e coluna(s) se aplicável:
        Formato: categoria|coluna1,coluna2
        Exemplo: correlation|Amount,Time
        """
        
        llm_classification = self._call_gemini(prompt, system_context)
        
        # PARTE 2: Mapeamento robusto (fallback para regras)
        query_lower = user_query.lower()
        
        # Parse da resposta da LLM
        try:
            if '|' in llm_classification:
                category, columns_str = llm_classification.split('|', 1)
                specific_columns = [col.strip() for col in columns_str.split(',') if col.strip()]
            else:
                category = llm_classification.strip()
                specific_columns = []
        except:
            category = "general"
            specific_columns = []
        
        # Fallback para regras se LLM falhar
        if category not in ['descriptive', 'correlation', 'distribution', 'outliers', 
                          'clustering', 'temporal', 'frequency', 'balance', 'insights', 'memory']:
            category = self._fallback_classification(query_lower)
            specific_columns = self._extract_columns_from_query(user_query, df)
        
        return {
            "category": category,
            "specific_columns": specific_columns,
            "original_query": user_query,
            "llm_interpretation": llm_classification,
            "confidence": "high" if '|' in llm_classification else "medium"
        }
    
    def generate_intelligent_response(self, query_result: Dict, analysis_result: Any, df: pd.DataFrame) -> str:
        """
        Gera resposta inteligente combinando resultados técnicos com insights da LLM
        """
        
        system_context = """Você é um consultor sênior em dados. 
        Transforme resultados técnicos em insights claros e actionables."""
        
        prompt = f"""
        Pergunta original: "{query_result['original_query']}"
        Tipo de análise: {query_result['category']}
        
        Resultados técnicos obtidos:
        {str(analysis_result)[:1000] if analysis_result else "Análise executada"}
        
        Dataset context: {self.dataset_context.get('dataset_type', 'Dataset genérico')}
        
        Gere uma resposta clara que:
        1. Responda diretamente à pergunta
        2. Explique os principais achados
        3. Forneça insights práticos
        4. Sugira próximos passos se relevante
        
        Use linguagem acessível e focada em valor de negócio.
        Máximo 300 palavras.
        """
        
        return self._call_gemini(prompt, system_context)
    
    def generate_smart_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Gera sugestões inteligentes baseadas no dataset"""
        
        system_context = """Gere 5 sugestões práticas de perguntas sobre análise de dados."""
        
        prompt = f"""
        Dataset: {self.dataset_context.get('dataset_type', 'Genérico')}
        Colunas: {list(df.columns)[:8]}
        Shape: {df.shape}
        
        Gere 5 perguntas específicas e práticas que um analista faria sobre este dataset.
        
        Formato: uma pergunta por linha, começando com "•"
        Exemplo:
        • Quais são as correlações mais fortes entre as variáveis numéricas?
        • Existem outliers significativos que precisam de investigação?
        """
        
        response = self._call_gemini(prompt, system_context)
        
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                suggestion = line[1:].strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:5]
    
    def generate_executive_conclusions(self, df: pd.DataFrame, memory: Dict) -> str:
        """Gera conclusões executivas com base na memória"""
        
        system_context = """Você é um CDO (Chief Data Officer) gerando conclusões executivas."""
        
        # Compilar histórico
        analyses_summary = []
        for conclusion in memory.get('conclusions', [])[-10:]:
            analyses_summary.append(f"- {conclusion.get('analysis_type', 'N/A')}: {conclusion.get('conclusion', 'N/A')[:100]}")
        
        prompt = f"""
        Dataset analisado: {df.shape[0]:,} registros, {df.shape[1]} variáveis
        Tipo: {self.dataset_context.get('dataset_type', 'Dataset genérico')}
        
        Análises realizadas:
        {chr(10).join(analyses_summary[:5])}
        
        Total de análises: {len(memory.get('conclusions', []))}
        
        Gere conclusões executivas estruturadas:
        
        ## 🎯 Resumo Executivo
        [Síntese em 2-3 frases]
        
        ## 🔍 Principais Descobertas  
        [3-4 descobertas mais importantes]
        
        ## 💼 Impacto no Negócio
        [Como isso afeta estratégia/operações]
        
        ## 🎯 Recomendações
        [3-4 ações específicas recomendadas]
        
        Seja conciso, objetivo e focado em valor executivo.
        """
        
        return self._call_gemini(prompt, system_context)
    
    # === MÉTODOS AUXILIARES ===
    
    def _extract_dataset_type(self, text: str) -> str:
        """Extrai tipo de dataset da resposta da LLM"""
        lines = text.lower().split('\n')
        for line in lines:
            if 'dataset' in line or 'tipo' in line:
                if 'fraude' in line:
                    return 'Detecção de Fraude'
                elif 'vendas' in line or 'sales' in line:
                    return 'Dados de Vendas'
                elif 'marketing' in line:
                    return 'Marketing Analytics'
                elif 'financeiro' in line or 'finance' in line:
                    return 'Dados Financeiros'
        return 'Dataset Genérico'
    
    def _extract_characteristics(self, text: str) -> List[str]:
        """Extrai características da resposta"""
        characteristics = []
        lines = text.split('\n')
        in_characteristics = False
        
        for line in lines:
            if 'característica' in line.lower() or 'principais' in line.lower():
                in_characteristics = True
                continue
            if in_characteristics and line.strip().startswith(('•', '-', '1.', '2.', '3.')):
                char = re.sub(r'^[•\-\d\.]\s*', '', line.strip())
                if char:
                    characteristics.append(char[:100])
                if len(characteristics) >= 3:
                    break
        
        return characteristics or ['Análise detalhada disponível']
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extrai recomendações da resposta"""
        recommendations = []
        lines = text.split('\n')
        in_recommendations = False
        
        for line in lines:
            if 'recomend' in line.lower() or 'análise' in line.lower():
                in_recommendations = True
                continue
            if in_recommendations and line.strip().startswith(('•', '-', '1.', '2.', '3.')):
                rec = re.sub(r'^[•\-\d\.]\s*', '', line.strip())
                if rec:
                    recommendations.append(rec[:100])
                if len(recommendations) >= 3:
                    break
        
        return recommendations or ['Análise exploratória', 'Detecção de padrões']
    
    def _fallback_classification(self, query_lower: str) -> str:
        """Classificação de fallback baseada em regras"""
        if any(word in query_lower for word in ['correlação', 'relaciona', 'correlation']):
            return 'correlation'
        elif any(word in query_lower for word in ['outlier', 'anomalia', 'atípico']):
            return 'outliers'
        elif any(word in query_lower for word in ['cluster', 'agrupamento', 'grupo']):
            return 'clustering'
        elif any(word in query_lower for word in ['distribuição', 'histograma']):
            return 'distribution'
        elif any(word in query_lower for word in ['frequente', 'comum', 'contagem']):
            return 'frequency'
        elif any(word in query_lower for word in ['tempo', 'temporal', 'trend']):
            return 'temporal'
        elif any(word in query_lower for word in ['balanceamento', 'balancear']):
            return 'balance'
        elif any(word in query_lower for word in ['conclusão', 'insight', 'descoberta']):
            return 'insights'
        elif any(word in query_lower for word in ['memória', 'histórico']):
            return 'memory'
        else:
            return 'descriptive'
    
    def _extract_columns_from_query(self, query: str, df: pd.DataFrame) -> List[str]:
        """Extrai nomes de colunas mencionadas na query"""
        columns_mentioned = []
        for col in df.columns:
            if col.lower() in query.lower():
                columns_mentioned.append(col)
        return columns_mentioned

# === FUNÇÕES ROBUSTAS DA VERSÃO 1 (MANTIDAS INTEGRALMENTE) ===

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
    
    summary += "\n📊 **Principais Descobertas:**\n"
    for i, conclusion in enumerate(memory['conclusions'][-5:], 1):
        summary += f"{i}. **{conclusion['analysis_type'].title()}:** {conclusion['conclusion'][:100]}...\n"
    
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

def perform_descriptive_analysis(df):
    """Análise descritiva robusta - VERSÃO 1 MANTIDA"""
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
    
    conclusion_text = f"Análise descritiva completa de {len(numeric_cols)} variáveis numéricas. " + "; ".join(conclusions[:3])
    add_to_memory("descriptive_analysis", conclusion_text, {
        "numeric_columns": len(numeric_cols),
        "total_rows": len(df),
        "total_columns": len(df.columns)
    })
    
    st.session_state.analysis_cache[cache_key] = desc_stats
    return desc_stats

def plot_correlation_heatmap(df):
    """Gera mapa de correlação robusto - VERSÃO 1 MANTIDA"""
    cache_key = "correlation_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None, "❌ Necessário pelo menos 2 colunas numéricas para correlação"
    
    fig, ax = plt.subplots(figsize=(min(16, max(8, len(numeric_cols))), min(12, max(6, len(numeric_cols)))))
    corr_matrix = df[numeric_cols].corr()
    
    # Usar máscara apenas se matriz for grande
    if len(numeric_cols) > 10:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                    center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
    else:
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', 
                    center=0, square=True, fmt='.2f', ax=ax, cbar_kws={"shrink": .8})
    
    ax.set_title('Matriz de Correlação - Análise Completa', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Análise das correlações
    high_corr_pairs = []
    moderate_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((col1, col2, corr_val))
                elif abs(corr_val) > 0.5:
                    moderate_corr_pairs.append((col1, col2, corr_val))
    
    conclusion = f"Análise de correlação entre {len(numeric_cols)} variáveis: "
    conclusion += f"{len(high_corr_pairs)} correlações altas (>0.7), "
    conclusion += f"{len(moderate_corr_pairs)} correlações moderadas (0.5-0.7)."
    
    if high_corr_pairs:
        top_corr = max(high_corr_pairs, key=lambda x: abs(x[2]))
        conclusion += f" Maior correlação: {top_corr[0]} ↔ {top_corr[1]} ({top_corr[2]:.3f})."
    
    add_to_memory("correlation_analysis", conclusion, {
        "high_correlations": len(high_corr_pairs),
        "moderate_correlations": len(moderate_corr_pairs),
        "variables_analyzed": len(numeric_cols)
    })
    
    st.session_state.agent_memory['generated_plots'].append({'analysis_type': 'correlation_analysis', 'figure': fig})
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def plot_distribution(df, column):
    """Gera histograma robusto - VERSÃO 1 MANTIDA"""
    cache_key = f"distribution_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma com KDE
    sns.histplot(df[column].dropna(), kde=True, ax=ax1)
    ax1.set_title(f'Distribuição de {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(y=df[column].dropna(), ax=ax2)
    ax2.set_title(f'Box Plot - {column}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Estatísticas
    data_clean = df[column].dropna()
    skewness = data_clean.skew()
    kurtosis = data_clean.kurtosis()
    q1, q3 = data_clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers_count = len(data_clean[(data_clean < q1 - 1.5*iqr) | (data_clean > q3 + 1.5*iqr)])
    
    if abs(skewness) > 1:
        skew_desc = "altamente assimétrica"
    elif abs(skewness) > 0.5:
        skew_desc = "moderadamente assimétrica"
    else:
        skew_desc = "aproximadamente simétrica"
    
    conclusion = f"Distribuição de '{column}': {skew_desc} (skew={skewness:.2f}), "
    conclusion += f"curtose={kurtosis:.2f}, {outliers_count} outliers detectados pelo método IQR"
    
    add_to_memory("distribution_analysis", conclusion, {
        "column": column,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "outliers_iqr": int(outliers_count)
    })
    
    st.session_state.agent_memory['generated_plots'].append({
        'analysis_type': 'distribution_analysis', 
        'figure': fig, 
        'column': column
    })
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def detect_outliers(df, column):
    """Detecção de outliers robusta - VERSÃO 1 MANTIDA"""
    cache_key = f"outlier_detection_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"❌ Coluna '{column}' não encontrada"
    
    data = df[[column]].dropna()
    
    if len(data) < 10:
        return None, f"❌ Poucos dados na coluna '{column}' para detecção de outliers"

    # Múltiplos métodos
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso_forest.fit_predict(data)
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
    
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    outliers_zscore = z_scores > 3
    
    outlier_count_iso = sum(1 for x in outliers_iso if x == -1)
    outlier_count_iqr = sum(outliers_iqr)
    outlier_count_zscore = sum(outliers_zscore)
    
    total_points = len(data)
    
    # Consenso entre métodos
    consensus_outliers = 0
    for i in range(len(data)):
        methods_agree = 0
        if outliers_iso[i] == -1:
            methods_agree += 1
        if outliers_iqr.iloc[i]:
            methods_agree += 1
        if outliers_zscore.iloc[i]:
            methods_agree += 1
        
        if methods_agree >= 2:
            consensus_outliers += 1
    
    conclusion = f"Análise de outliers em '{column}': "
    conclusion += f"Isolation Forest: {outlier_count_iso} ({outlier_count_iso/total_points*100:.1f}%), "
    conclusion += f"IQR: {outlier_count_iqr} ({outlier_count_iqr/total_points*100:.1f}%), "
    conclusion += f"Z-score: {outlier_count_zscore} ({outlier_count_zscore/total_points*100:.1f}%). "
    conclusion += f"Consenso (≥2 métodos): {consensus_outliers} outliers"
    
    add_to_memory("outlier_detection", conclusion, {
        "column": column,
        "isolation_forest": outlier_count_iso,
        "iqr_method": outlier_count_iqr,
        "zscore_method": outlier_count_zscore,
        "consensus_outliers": consensus_outliers,
        "total_points": total_points
    })
    
    st.session_state.analysis_cache[cache_key] = (None, conclusion)
    return None, conclusion

def perform_clustering_analysis(df, n_clusters=None, sample_size=5000):
    """Análise de clustering robusta - VERSÃO 1 MANTIDA"""
    cache_key = "clustering_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "❌ Necessário pelo menos 2 colunas numéricas para clustering"
    
    actual_sample_size = min(sample_size, len(df))
    
    # Amostragem inteligente
    if len(df) > sample_size:
        df_sample = df.sample(n=actual_sample_size, random_state=42)
    else:
        df_sample = df
    
    df_numeric = df_sample[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric.fillna(df_numeric.mean()))
    
    if n_clusters is None:
        silhouette_scores = []
        K_range = range(2, min(11, len(df_sample)//50 + 2))
        
        if len(K_range) < 1:
            return None, "❌ Amostra muito pequena para clustering"

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        n_clusters = K_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
    else:
        best_silhouette = None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    if best_silhouette is None:
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    else:
        silhouette_avg = best_silhouette
    
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
    
    if silhouette_avg > 0.7:
        quality_desc = "excelente separação"
    elif silhouette_avg > 0.5:
        quality_desc = "boa separação"
    elif silhouette_avg > 0.3:
        quality_desc = "separação moderada"
    else:
        quality_desc = "separação fraca"
    
    conclusion = f"Clustering K-means identificou {n_clusters} grupos com {quality_desc} "
    conclusion += f"(Silhouette: {silhouette_avg:.3f}). "
    conclusion += f"Maior cluster: {max(cluster_sizes)} pontos, menor: {min(cluster_sizes)} pontos"
    
    add_to_memory("clustering_analysis", conclusion, {
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_avg),
        "cluster_sizes": cluster_sizes,
        "sample_size": actual_sample_size
    })
    
    st.session_state.analysis_cache[cache_key] = (None, conclusion)
    return None, conclusion

def analyze_frequent_values(df, max_categories=10):
    """Análise de frequência robusta - VERSÃO 1 MANTIDA"""
    cache_key = "frequency_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    discrete_cols = [col for col in df.select_dtypes(include=['int64']).columns 
                    if df[col].nunique() <= 20]
    
    results = {}
    conclusions = []
    
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
            
            most_freq_val = value_counts.index[0]
            most_freq_count = value_counts.iloc[0]
            conclusions.append(f"'{col}': mais frequente = '{most_freq_val}' ({most_freq_count}x)")
    
    for col in discrete_cols:
        if col not in categorical_cols:  # Evitar duplicatas
            value_counts = df[col].value_counts()
            results[col] = {
                'type': 'discrete',
                'most_frequent': value_counts.head(3).to_dict(),
                'least_frequent': value_counts.tail(3).to_dict(),
                'unique_count': df[col].nunique(),
                'null_count': df[col].isnull().sum()
            }
            
            most_freq_val = value_counts.index[0]
            most_freq_count = value_counts.iloc[0]
            conclusions.append(f"'{col}': mais frequente = {most_freq_val} ({most_freq_count}x)")
    
    if conclusions:
        conclusion_text = f"Análise de frequência: {len(results)} colunas analisadas. " + "; ".join(conclusions[:3])
        add_to_memory("frequency_analysis", conclusion_text, {"analyzed_columns": len(results)})
    
    st.session_state.analysis_cache[cache_key] = results
    return results

def analyze_balance(df, column):
    """Análise de balanceamento robusta - VERSÃO 1 MANTIDA"""
    cache_key = f"balance_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"❌ Coluna '{column}' não encontrada"
    
    unique_values = df[column].nunique()
    if unique_values != 2:
        return None, f"❌ A coluna '{column}' não é binária (possui {unique_values} valores únicos)"
    
    value_counts = df[column].value_counts()
    total_count = len(df[column])
    
    fig = plt.figure(figsize=(8, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
           startangle=90, colors=colors)
    plt.title(f'Balanceamento da Coluna {column}')
    plt.axis('equal')
    plt.tight_layout()
    
    conclusion = f"Balanceamento da coluna '{column}': "
    for val, count in value_counts.items():
        percentage = (count / total_count) * 100
        conclusion += f"Valor {val}: {count:,} ({percentage:.1f}%); "
    
    if value_counts.min() / total_count < 0.10:
        conclusion += "Dataset altamente desbalanceado"
    elif value_counts.min() / total_count < 0.30:
        conclusion += "Dataset moderadamente desbalanceado"
    else:
        conclusion += "Dataset relativamente balanceado"
    
    add_to_memory("balance_analysis", conclusion, {
        "column": column,
        "value_counts": value_counts.to_dict(),
        "total_count": total_count
    })
    
    st.session_state.agent_memory['generated_plots'].append({
        'analysis_type': 'balance_analysis', 
        'figure': fig, 
        'column': column
    })
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def analyze_temporal_patterns(df, time_column):
    """Análise temporal robusta - VERSÃO 1 MANTIDA"""
    cache_key = f"temporal_analysis_{time_column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if time_column not in df.columns:
        return None, f"Coluna '{time_column}' não encontrada"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histograma temporal
    ax1.hist(df[time_column].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Distribuição Temporal - {time_column}')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    # Padrões por classe (se existir coluna Class)
    if 'Class' in df.columns:
        for class_val in sorted(df['Class'].unique()):
            subset = df[df['Class'] == class_val]
            ax2.hist(subset[time_column].dropna(), bins=30, alpha=0.6, 
                    label=f'Classe {class_val} (n={len(subset)})')
        ax2.set_title('Padrões Temporais por Classe')
        ax2.legend()
    else:
        # Linha temporal geral
        time_sorted = df.sort_values(time_column)
        ax2.plot(time_sorted[time_column], range(len(time_sorted)), alpha=0.7)
        ax2.set_title('Evolução Temporal')
    
    ax2.set_xlabel('Tempo')
    ax2.grid(True, alpha=0.3)
    
    # Intervalos temporais
    time_diffs = df[time_column].diff().dropna()
    if len(time_diffs) > 0:
        ax3.hist(time_diffs, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Distribuição de Intervalos Temporais')
        ax3.set_xlabel('Diferença de Tempo')
        ax3.set_ylabel('Frequência')
    ax3.grid(True, alpha=0.3)

    # Box plot temporal
    bp = ax4.boxplot(df[time_column].dropna(), patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    ax4.set_title(f'Box Plot - {time_column}')
    ax4.set_ylabel('Tempo')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Estatísticas temporais
    time_data = df[time_column].dropna()
    time_range = time_data.max() - time_data.min()
    time_mean = time_data.mean()
    time_std = time_data.std()
    
    conclusion = f"Análise temporal de {len(time_data)} registros em {time_range:.0f} unidades de tempo. "
    conclusion += f"Média: {time_mean:.0f}, desvio: {time_std:.0f}"

    add_to_memory("temporal_analysis", conclusion, {
        "time_column": time_column,
        "time_range": float(time_range),
        "time_mean": float(time_mean),
        "records_analyzed": len(time_data)
    })

    st.session_state.agent_memory['generated_plots'].append({
        'analysis_type': 'temporal_analysis', 
        'figure': fig, 
        'column': time_column
    })
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def generate_complete_analysis_summary(df):
    """Gera resumo completo - VERSÃO 1 MANTIDA"""
    summary_text = f"""
🤖 **RESUMO COMPLETO DAS ANÁLISES - AGENTE HÍBRIDO COM GEMINI**

{get_dataset_info(df)}

📈 **ANÁLISES REALIZADAS:**
"""
    
    for i, conclusion_entry in enumerate(st.session_state.agent_memory['conclusions'], 1):
        analysis_name = conclusion_entry['analysis_type'].replace('_', ' ').title()
        summary_text += f"\n**{i}. {analysis_name}:**\n"
        summary_text += f"   {conclusion_entry['conclusion']}\n"
        summary_text += f"   *{conclusion_entry['timestamp'][:19]}*\n"
    
    summary_text += f"""

🎯 **INSIGHTS PRINCIPAIS:**

1. **Qualidade:** Dataset com {df.shape[0]:,} registros e {df.shape[1]} variáveis
2. **Completude:** {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%
3. **Análises:** {len(st.session_state.agent_memory['conclusions'])} análises executadas
4. **Inteligência:** Agente híbrido com LLM + funções robustas

📊 **TOTAL DE ANÁLISES:** {len(st.session_state.agent_memory['conclusions'])}
🧠 **POWERED BY:** Gemini + Funções Robustas
🕐 **GERADO EM:** {datetime.now().strftime("%d/%m/%Y às %H:%M:%S")}

---
🤖 **Agente Híbrido - I2A2 Academy 2025**
"""
    return summary_text

def generate_pdf_report(df, gemini_insights=None):
    """Gera PDF robusto - VERSÃO 1 APRIMORADA"""
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Página 1: Capa
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.95, '🤖 RELATÓRIO AGENTE HÍBRIDO', 
                ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.90, 'Powered by Gemini + Funções Robustas', 
                ha='center', va='top', fontsize=12, style='italic')
        ax.text(0.5, 0.87, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                ha='center', va='top', fontsize=10)
        
        # Informações do dataset
        dataset_info = get_dataset_info(df)
        ax.text(0.05, 0.75, dataset_info, ha='left', va='top', fontsize=8, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd"))
        
        # Insights do Gemini se disponível
        if gemini_insights:
            ax.text(0.05, 0.40, "🧠 INSIGHTS DA IA:", ha='left', va='top', 
                   fontsize=10, fontweight='bold')
            ax.text(0.05, 0.35, gemini_insights[:500] + "...", ha='left', va='top', 
                   fontsize=8, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e5f5"))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Página 2+: Resumo das análises
        analysis_summary = generate_complete_analysis_summary(df)
        
        # Dividir texto em páginas
        max_chars_per_page = 3000
        text_chunks = [analysis_summary[i:i+max_chars_per_page] 
                      for i in range(0, len(analysis_summary), max_chars_per_page)]

        for i, chunk in enumerate(text_chunks):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.05, 0.95, f'RESUMO DAS ANÁLISES (Página {i+1})', 
                   ha='left', va='top', fontsize=14, fontweight='bold')
            ax.text(0.05, 0.90, chunk, ha='left', va='top', fontsize=9, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Adicionar gráficos
        for plot_data in st.session_state.agent_memory.get('generated_plots', []):
            fig = plot_data.get('figure')
            if fig:
                # Adicionar título descritivo
                analysis_type = plot_data.get('analysis_type', 'Análise').replace('_', ' ').title()
                column = plot_data.get('column', '')
                title = f'{analysis_type}' + (f' - {column}' if column else '')
                fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# === FUNÇÕES DE COMPATIBILIDADE ===

def interpret_question(question, df):
    """Função de compatibilidade - agora usa HybridGeminiAgent"""
    # Esta função é mantida para compatibilidade com a versão 1
    # Mas na prática, usaremos o método interpret_query_intelligently
    return "general"

def get_adaptive_suggestions(df):
    """Função de compatibilidade - agora usa HybridGeminiAgent"""
    # Sugestões básicas como fallback
    suggestions = [
        "• Mostre estatísticas descritivas",
        "• Mostre correlações entre variáveis", 
        "• Faça clustering automático dos dados",
        "• Detecte outliers nas colunas numéricas",
        "• Mostre valores mais frequentes",
        "• Qual sua memória de análises?"
    ]
    return suggestions
