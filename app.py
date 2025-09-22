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
    st.info("👆 Faça upload de um CSV para começar.")
