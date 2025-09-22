import streamlit as st
import pandas as pd
from utils import (
    perform_descriptive_analysis, plot_distribution, plot_correlation_heatmap,
    analyze_temporal_patterns, perform_clustering_analysis, detect_outliers,
    analyze_frequent_values, generate_pdf_report, get_memory_summary,
    interpret_question, get_adaptive_suggestions
)

st.set_page_config(page_title="Agente de Análise de Dados", page_icon="🤖", layout="wide")

# Inicializa memória
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = {'conclusions': [], 'insights': [], 'patterns_found': [], 'analysis_history': [], 'generated_plots': []}
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

st.title("🤖 Agente Autônomo de Análise de Dados")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    st.dataframe(df.head())

    user_question = st.text_input("Digite sua pergunta")
    if user_question:
        analysis_type = interpret_question(user_question, df)
        st.info(f"🔍 Pergunta interpretada como: {analysis_type}")

        if analysis_type == "descriptive":
            st.dataframe(perform_descriptive_analysis(df))
        elif analysis_type == "correlation":
            st.pyplot(plot_correlation_heatmap(df))
        elif analysis_type == "outliers":
            col = st.selectbox("Coluna para outliers", df.select_dtypes(include="number").columns)
            _, msg = detect_outliers(df, col)
            st.info(msg)
        elif analysis_type == "temporal":
            time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_cols:
                st.pyplot(analyze_temporal_patterns(df, time_cols[0]))
            else:
                st.warning("Nenhuma coluna temporal encontrada")
        elif analysis_type == "clustering":
            _, msg = perform_clustering_analysis(df)
            st.info(msg)

    st.markdown("💡 Sugestões de Perguntas:")
    for s in get_adaptive_suggestions(df):
        st.markdown(s)

    if st.button("📄 Gerar Relatório PDF"):
        pdf = generate_pdf_report(df)
        st.download_button("Download PDF", pdf, "relatorio.pdf", "application/pdf")

else:
    # Página inicial
    st.info("👆 **Faça upload de um arquivo CSV na barra lateral para começar a análise inteligente.**")

    st.markdown("""
    ## 🚀 **Agente Autônomo de Análise de Dados Inteligente e Adaptativa**

    ### 🧠 **Funcionalidades Inteligentes:**

    #### **💡 Memória Persistente**
    - 🔄 Armazena todas as conclusões de análises realizadas
    - 📚 Histórico completo de insights descobertos
    - 🤖 Capacidade de consultar análises anteriores
    - 📊 Resumos inteligentes das descobertas

    #### **📈 Análises Avançadas Disponíveis:**
    - ✅ **Estatísticas Descritivas:** Análise completa com insights automáticos
    - ✅ **Correlações:** Mapas de calor com identificação de relações significativas
    - ✅ **Padrões Temporais:** Detecção de tendências e comportamentos ao longo do tempo
    - ✅ **Clustering Inteligente:** Agrupamento automático com otimização de parâmetros
    - ✅ **Detecção de Outliers:** Múltiplos métodos (Isolation Forest, IQR, Z-score)
    - ✅ **Análise de Frequências:** Valores mais/menos comuns em dados categóricos

    #### **🎯 Recursos Especiais:**
    - 🤖 **Interpretação de Linguagem Natural Aprimorada:** Compreende perguntas em português
    - 📄 **Exportação PDF:** Gera relatórios profissionais automaticamente
    - ⚡ **Resumo Completo:** Executa e compila todas as análises em texto
    - 🎨 **Interface Otimizada:** Foco na usabilidade e resultados
    - 🔧 **Otimização de Performance:** Amostragem inteligente para datasets grandes
    - 🧠 **Inteligência Adaptativa:** Sugere análises com base na estrutura do dataset

    ### 📝 **Exemplos de Perguntas Inteligentes:**

    - *"Mostre correlações entre variáveis"*
    - *"Mostre padrões temporais nos dados"*
    - *"Faça clustering automático dos dados"*
    - *"Detecte outliers com múltiplos métodos"*
    - *"Mostre valores mais frequentes"*
    - *"Qual sua memória de análises?"*

    ---


    ✅ **Memória do Agente** | ✅ **Padrões Temporais** | ✅ **Clustering** | ✅ **Exportação PDF** | ✅ **Inteligência Adaptativa**
    """)


# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
🤖 <strong>Agente Autônomo de Análise de Dados</strong><br>
Desenvolvido para o <strong>Desafio I2A2 Academy</strong> por Ana Manuella Ribeiro | Setembro 2025<br>
</div>
""", unsafe_allow_html=True)



