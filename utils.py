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
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
import base64
import re
import streamlit as st # Necessário para st.session_state

# --- FUNÇÕES DE GERENCIAMENTO DE MEMÓRIA E CACHE ---

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

def generate_complete_analysis_summary(df):
    """Gera resumo completo de todas as análises realizadas, garantindo que não haja repetições."""
    summary_text = f"""
🤖 **RESUMO COMPLETO DAS ANÁLISES - AGENTE AUTÔNOMO**

{get_dataset_info(df)}

📈 **ANÁLISES REALIZADAS:**

"""
    
    # Compilar todas as conclusões da memória
    for i, conclusion_entry in enumerate(st.session_state.agent_memory['conclusions'], 1):
        analysis_name = conclusion_entry['analysis_type'].replace('_', ' ').title()
        summary_text += f"\n**{i}. {analysis_name}:**\n"
        summary_text += f"   {conclusion_entry['conclusion']}\n"
        summary_text += f"   *Realizada em: {conclusion_entry['timestamp'][:19]}*\n"
    
    summary_text += f"""

🎯 **INSIGHTS PRINCIPAIS:**

1. **Qualidade dos Dados:** Dataset com {df.shape[0]:,} registros e {df.shape[1]} variáveis, apresentando {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}% de completude.

2. **Estrutura:** Composto por {len(df.select_dtypes(include=[np.number]).columns)} variáveis numéricas e {len(df.select_dtypes(include=['object', 'category']).columns)} categóricas.

3. **Padrões Identificados:** {len(st.session_state.agent_memory['conclusions'])} análises diferentes foram executadas, revelando padrões em distribuições, correlações e agrupamentos.

4. **Recomendações:** Com base nas análises realizadas, o dataset apresenta características adequadas para modelagem preditiva e análise exploratória avançada.

📊 **TOTAL DE ANÁLISES EXECUTADAS:** {len(st.session_state.agent_memory['conclusions'])}
🕐 **GERADO EM:** {datetime.now().strftime("%d/%m/%Y às %H:%M:%S")}

---
🤖 **Agente Autônomo de Análise de Dados - I2A2 Academy 2025**
"""
    
    return summary_text

# --- FUNÇÕES DE ANÁLISE DE DADOS ---

def analyze_frequent_values(df, max_categories=10):
    """Análise de valores mais/menos frequentes."""
    cache_key = "frequency_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    discrete_cols = [col for col in df.select_dtypes(include=['int64']).columns 
                    if df[col].nunique() <= 20 and col not in categorical_cols]
    
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
            conclusions.append(f"Coluna '{col}': valor mais frequente é '{most_freq_val}' ({most_freq_count} ocorrências)")
    
    for col in discrete_cols:
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
        conclusions.append(f"Coluna '{col}': valor mais frequente é {most_freq_val} ({most_freq_count} ocorrências)")
    
    if conclusions:
        conclusion_text = f"Análise de frequência: {len(results)} colunas analisadas. " + "; ".join(conclusions[:3])
        add_to_memory("frequency_analysis", conclusion_text, {"analyzed_columns": len(results)})
    
    st.session_state.analysis_cache[cache_key] = results
    return results

def perform_descriptive_analysis(df):
    """Análise descritiva aprimorada com insights automáticos."""
    cache_key = "descriptive_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    desc_stats = df.describe()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    conclusions = []
    
    for col in numeric_cols[:5]: # Limitar para não sobrecarregar a memória
        mean_val = df[col].mean()
        std_val = df[col].std()
        median_val = df[col].median()
        
        if std_val == 0: # Evitar divisão por zero
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

def plot_distribution(df, column):
    """Gera histograma aprimorado com análise estatística detalhada."""
    cache_key = f"distribution_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df[column], kde=True, ax=ax1)
    ax1.set_title(f'Distribuição de {column}')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    sns.boxplot(y=df[column], ax=ax2)
    ax2.set_title(f'Box Plot - {column}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers_count = len(df[(df[column] < q1 - 1.5*iqr) | (df[column] > q3 + 1.5*iqr)])
    
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
    
    st.session_state.agent_memory['generated_plots'].append({'analysis_type': 'distribution_analysis', 'figure': fig, 'column': column})
    st.session_state.analysis_cache[cache_key] = fig
    return fig, conclusion # Retorna a figura e a conclusão

def plot_correlation_heatmap(df):
    """Gera mapa de correlação aprimorado com análise de significância."""
    cache_key = "correlation_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None, "❌ Necessário pelo menos 2 colunas numéricas para clustering"
    
    fig, ax = plt.subplots(figsize=(16, 12))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                center=0, square=True, fmt='.2f', ax=ax,
                cbar_kws={"shrink": .8})
    ax.set_title('Matriz de Correlação - Análise Completa', fontsize=16, pad=20)
    
    high_corr_pairs = []
    moderate_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((col1, col2, corr_val))
            elif abs(corr_val) > 0.5:
                moderate_corr_pairs.append((col1, col2, corr_val))
    
    conclusion = f"Análise de correlação entre {len(numeric_cols)} variáveis: "
    conclusion += f"{len(high_corr_pairs)} correlações altas (>0.7), "
    conclusion += f"{len(moderate_corr_pairs)} correlações moderadas (0.5-0.7). "
    
    if high_corr_pairs:
        top_corr = max(high_corr_pairs, key=lambda x: abs(x[2]))
        conclusion += f"Maior correlação: {top_corr[0]} ↔ {top_corr[1]} ({top_corr[2]:.3f})"
    
    add_to_memory("correlation_analysis", conclusion, {
        "high_correlations": len(high_corr_pairs),
        "moderate_correlations": len(moderate_corr_pairs),
        "variables_analyzed": len(numeric_cols)
    })
    
    st.session_state.agent_memory['generated_plots'].append({'analysis_type': 'correlation_analysis', 'figure': fig})
    st.session_state.analysis_cache[cache_key] = (fig, conclusion) # Cacheia a figura e a conclusão
    return fig, conclusion # Retorna a figura e a conclusão

def analyze_temporal_patterns(df, time_column):
    """Análise de padrões temporais aprimorada com detecção de tendências."""
    cache_key = f"temporal_analysis_{time_column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if time_column not in df.columns:
        return None, "Coluna '{time_column}' não encontrada no dataset"
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1.hist(df[time_column], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Distribuição Temporal - {time_column}')
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    if 'Class' in df.columns: # Exemplo de adaptação para coluna 'Class'
        for class_val in sorted(df['Class'].unique()):
            subset = df[df['Class'] == class_val]
            ax2.hist(subset[time_column], bins=30, alpha=0.6, 
                    label=f'Classe {class_val} (n={len(subset)})')
        ax2.set_title('Padrões Temporais por Classe')
        ax2.set_xlabel('Tempo')
        ax2.set_ylabel('Frequência')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        time_sorted = df.sort_values(time_column)
        ax2.plot(time_sorted[time_column], range(len(time_sorted)), alpha=0.7, color='orange')
        ax2.set_title('Tendência Temporal dos Dados')
        ax2.set_xlabel('Tempo')
        ax2.set_ylabel('Ordem dos Registros')
        ax2.grid(True, alpha=0.3)

    time_diffs = df[time_column].diff().dropna()
    ax3.hist(time_diffs, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Distribuição de Intervalos Temporais')
    ax3.set_xlabel('Diferença de Tempo')
    ax3.set_ylabel('Frequência')
    ax3.grid(True, alpha=0.3)

    ax4.boxplot(df[time_column])
    ax4.set_title(f'Box Plot - {time_column}')
    ax4.set_ylabel('Tempo')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    time_range = df[time_column].max() - df[time_column].min()
    time_mean = df[time_column].mean()
    time_std = df[time_column].std()
    time_median = df[time_column].median()
    concentration_ratio = time_std / time_mean if time_mean != 0 else 0

    if concentration_ratio < 0.3:
        temporal_pattern = "dados altamente concentrados temporalmente"
    elif concentration_ratio < 0.7:
        temporal_pattern = "dados moderadamente distribuídos no tempo"
    else:
        temporal_pattern = "dados amplamente distribuídos ao longo do tempo"

    if abs(time_mean - time_median) > time_std * 0.5:
        trend_desc = "com tendência assimétrica"
    else:
        trend_desc = "com distribuição temporal equilibrada"

    conclusion = f"Análise temporal de {len(df)} registros: período de {time_range:.0f} unidades, "
    conclusion += f"{temporal_pattern} {trend_desc}. "
    conclusion += f"Tempo médio: {time_mean:.0f}, mediana: {time_median:.0f}"

    add_to_memory("temporal_analysis", conclusion, {
        "time_column": time_column,
        "time_range": float(time_range),
        "time_mean": float(time_mean),
        "concentration_ratio": float(concentration_ratio),
        "records_analyzed": len(df)
    })

    st.session_state.agent_memory['generated_plots'].append({'analysis_type': 'temporal_analysis', 'figure': fig, 'column': time_column})
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def perform_clustering_analysis(df, n_clusters=None, sample_size=None):
    """Análise de clustering aprimorada com otimização automática."""
    cache_key = "clustering_analysis"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "❌ Necessário pelo menos 2 colunas numéricas para clustering"
    
    if sample_size is None:
        sample_size = st.session_state.max_sample_size # Usar o valor do slider
    
    actual_sample_size = min(sample_size, len(df))
    
    # Amostragem inteligente para datasets grandes
    if len(df) > sample_size:
        if 'Class' in df.columns and df['Class'].nunique() == 2: # Se for binário e desbalanceado
            df_sample = df.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // df['Class'].nunique()), random_state=42)
            )
        else:
            df_sample = df.sample(n=actual_sample_size, random_state=42)
    else:
        df_sample = df
    
    df_numeric = df_sample[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric.fillna(df_numeric.mean()))
    
    if n_clusters is None:
        silhouette_scores = []
        K_range = range(2, min(11, len(df_sample)//50 + 2)) # Evitar muitos clusters para amostras pequenas
        
        if len(K_range) < 1: # Garantir que K_range não esteja vazio
            return None, "❌ Amostra muito pequena para clustering significativo."

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
    conclusion += f"Maior cluster: {max(cluster_sizes)} pontos, menor: {min(cluster_sizes)} pontos. "
    conclusion += f"Análise baseada em amostra de {actual_sample_size} registros"
    
    add_to_memory("clustering_analysis", conclusion, {
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_avg),
        "cluster_sizes": cluster_sizes,
        "sample_size": actual_sample_size,
        "quality_rating": quality_desc
    })
    
    st.session_state.analysis_cache[cache_key] = (None, conclusion) # Cacheia o resultado da análise
    return None, conclusion  # Retornando None para figura para simplificar

def detect_outliers(df, column):
    """Detecção de outliers aprimorada com múltiplos métodos."""
    cache_key = f"outlier_detection_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"❌ Coluna '{column}' não encontrada"
    
    data = df[[column]].dropna()
    
    if len(data) < 10: # Mínimo de dados para IsolationForest
        return None, f"❌ Poucos dados na coluna '{column}' para detecção de outliers."

    iso_forest = IsolationForest(contamination=st.session_state.contamination_rate, random_state=42)
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

def analyze_balance(df, column):
    """Análise de balanceamento para colunas binárias."""
    cache_key = f"balance_analysis_{column}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]

    if column not in df.columns:
        return None, f"❌ Coluna '{column}' não encontrada"
    
    unique_values = df[column].nunique()
    if unique_values != 2:
        return None, f"❌ A coluna '{column}' não é binária (possui {unique_values} valores únicos)."
    
    value_counts = df[column].value_counts()
    total_count = len(df[column])
    
    fig = plt.figure(figsize=(8, 6))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    plt.title(f'Balanceamento da Coluna {column}')
    plt.axis('equal') # Garante que o gráfico de pizza seja um círculo.
    
    conclusion = f"Análise de balanceamento da coluna '{column}': "
    for val, count in value_counts.items():
        percentage = (count / total_count) * 100
        conclusion += f"Valor {val}: {count:,} ({percentage:.2f}%); "
    
    if value_counts.min() / total_count < 0.10:
        conclusion += "Dataset altamente desbalanceado."
    else:
        conclusion += "Dataset relativamente balanceado."
    
    add_to_memory("balance_analysis", conclusion, {
        "column": column,
        "value_counts": value_counts.to_dict(),
        "total_count": total_count
    })
    
    st.session_state.agent_memory['generated_plots'].append({'analysis_type': 'balance_analysis', 'figure': fig, 'column': column})
    st.session_state.analysis_cache[cache_key] = (fig, conclusion)
    return fig, conclusion

def generate_pdf_report(df):
    """Gera relatório PDF com todas as análises e informações do dataset."""
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Página 1: Capa e informações do dataset
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100) # A4 size in inches
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.95, '🤖 RELATÓRIO COMPLETO DE ANÁLISE DE DADOS', 
                ha='center', va='top', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.90, 'Agente Autônomo de Análise', 
                ha='center', va='top', fontsize=12)
        ax.text(0.5, 0.87, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                ha='center', va='top', fontsize=10)
        
        # Informações do dataset
        dataset_info = get_dataset_info(df)
        ax.text(0.05, 0.80, dataset_info, ha='left', va='top', fontsize=8, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0f7fa", edgecolor="#00bcd4"))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Página 2 em diante: Resumo das análises e gráficos
        analysis_summary_full = generate_complete_analysis_summary(df)
        
        # Dividir o texto do resumo em chunks para caber nas páginas
        chunks = []
        current_chunk = ""
        max_chars_per_page = 2500 # Aumentar para caber mais texto por página
        
        # Adicionar título para o resumo das análises
        summary_lines = analysis_summary_full.split('\n')
        summary_title = summary_lines[0] # Assume que a primeira linha é o título
        summary_content = '\n'.join(summary_lines[1:])

        # Processar o conteúdo do resumo
        for line in summary_content.split('\n'):
            if len(current_chunk) + len(line) < max_chars_per_page:
                current_chunk += line + '\n'
            else:
                chunks.append(current_chunk)
                current_chunk = line + '\n'
        if current_chunk:
            chunks.append(current_chunk)

        for i, chunk in enumerate(chunks):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
            ax = fig.add_subplot(111)
            ax.text(0.05, 0.95, f'{summary_title} (Página {i+1})', ha='left', va='top', fontsize=14, fontweight='bold')
            ax.text(0.05, 0.90, chunk, ha='left', va='top', fontsize=8, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Adicionar gráficos gerados
        for plot_data in st.session_state.agent_memory['generated_plots']:
            fig = plot_data['figure']
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- INTERPRETAÇÃO DE PERGUNTAS E SUGESTÕES ADAPTATIVAS ---

def interpret_question(question, df):
    """Interpretação semântica avançada de perguntas do usuário."""
    question_lower = question.lower()
    
    # Mapeamento de palavras-chave 
    keywords_map = {
        'correlation': [r'correlaç(ão|ões)', r'relaç(ão|ões)', 'correlacionar', 'associação', 'dependência'],
        'temporal': ['tempo', 'temporal', 'tendência', 'padrão temporal', 'série', 'cronológico', 'data'],
        'clustering': ['cluster', 'agrupamento', 'grupo', 'segmentação', 'partição'],
        'outliers': ['outlier', 'outliers', 'anomalia', 'anômalo', 'atípico', 'discrepante', 'aberrante'],
        'frequency': ['frequente', 'frequentes', 'comum', 'raro', 'valores', 'contagem', 'ocorrência'],
        'memory': ['memória', 'conclusão', 'histórico', 'resumo', 'descobertas', 'insights'],
        'export': ['exportar', 'pdf', 'relatório', 'salvar', 'download'],
        'descriptive': ['estatística', 'estatísticas', 'resumo', 'descritiva', 'média', 'mediana', 'desvio'],
        'distribution': ['distribuição', 'histograma', 'frequência', 'densidade', 'spread'],
        'balance': ['balanceamento', 'balancear', 'proporção', 'desbalanceado'] # Adicionado para balanceamento
    }
    
    # Verificar palavras-chave específicas 
    for analysis_type, keywords in keywords_map.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', question_lower): # Usar regex para palavras inteiras
                return analysis_type
    
    # Análise de verbos apenas se não encontrou palavras-chave específicas
    if any(verb in question_lower for verb in ['mostre', 'exiba', 'visualize', 'plote', 'faça', 'analise']):
        # Análise mais específica para verbos
        if any(word in question_lower for word in ['tempo', 'temporal', 'data']):
            return 'temporal'
        elif any(word in question_lower for word in ['correlação', 'correlações', 'relação']):
            return 'correlation'
        elif any(word in question_lower for word in ['cluster', 'agrupamento']):
            return 'clustering'
        elif any(word in question_lower for word in ['outlier', 'anomalia']):
            return 'outliers'
        elif any(word in question_lower for word in ['frequente', 'frequentes', 'contagem']):
            return 'frequency'
        elif any(word in question_lower for word in ['estatística', 'descritiva', 'resumo']):
            return 'descriptive'
        elif any(word in question_lower for word in ['balanceamento', 'balancear', 'proporção']):
            return 'balance'
        else:
            return 'distribution'  # Default para verbos genéricos
    
    return 'general'

def get_adaptive_suggestions(df):
    """Gera sugestões de análise adaptativas baseadas na estrutura do dataset."""
    suggestions = []
    
    # 1. Colunas Numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        suggestions.append('• "Mostre estatísticas descritivas" - Resumo das variáveis numéricas')
        if len(numeric_cols) > 1:
            suggestions.append('• "Mostre correlações entre variáveis" - Relação entre colunas numéricas')
        suggestions.append('• "Detecte outliers com múltiplos métodos" - Encontra valores anômalos')
        # Não sugerir distribuição para todas as colunas, apenas como exemplo
        # suggestions.append('• "Qual a distribuição da coluna [Nome da Coluna]?" - Ex: Qual a distribuição da coluna Amount?')
    
    # 2. Colunas Categóricas/Discretas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    discrete_int_cols = [col for col in df.select_dtypes(include=['int64']).columns if df[col].nunique() <= 20]
    if len(categorical_cols) > 0 or len(discrete_int_cols) > 0:
        suggestions.append('• "Mostre valores mais frequentes" - Análise de distribuição de categorias/discretas')
    
    # 3. Colunas Temporais
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if len(time_cols) > 0:
        suggestions.append('• "Mostre padrões temporais nos dados" - Análise de tendências ao longo do tempo')
    
    # 4. Colunas Binárias Desbalanceadas (ex: 'Class')
    for col in df.columns:
        if df[col].nunique() == 2:
            counts = df[col].value_counts(normalize=True)
            if counts.min() < 0.10: # Se a menor classe for menos de 10%
                suggestions.append(f'• "Analise o balanceamento da coluna {col}"')
    
    # 5. Clustering (se houver dados numéricos suficientes)
    if len(numeric_cols) >= 2:
        suggestions.append('• "Faça clustering automático dos dados" - Identifica grupos naturais nos dados')
    
    # 6. Memória do Agente
    suggestions.append('• "Qual sua memória de análises?" - Consulta histórico de descobertas')
    
    return suggestions[:6] # Limitar a 6 sugestões

