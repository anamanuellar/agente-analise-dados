import streamlit as st
import pandas as pd
import io
import base64
from utils import (
    GeminiAgent,
    add_to_memory,
    get_memory_summary,
    get_dataset_info,
    analyze_frequent_values,
    perform_descriptive_analysis,
    plot_distribution,
    plot_correlation_heatmap,
    perform_clustering_analysis,
    detect_outliers,
    analyze_balance,
    generate_pdf_report
)

# ==========================
# CONFIGURAÇÃO DA PÁGINA
# ==========================
st.set_page_config(
    page_title="Agente de Análise de Dados com LLM",
    page_icon="🤖",
    layout="wide"
)

# ==========================
# ESTADO GLOBAL
# ==========================
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {
        "conclusions": [],
        "insights": [],
        "patterns_found": [],
        "analysis_history": [],
        "generated_plots": [],
        "llm_interactions": []
    }

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

if "gemini_agent" not in st.session_state:
    st.session_state.gemini_agent = GeminiAgent()

# ==========================
# INTERFACE
# ==========================
st.title("🤖 Agente Autônomo de Análise de Dados com LLM")
st.markdown("*Powered by Google Gemini + Estatística Tradicional*")
st.markdown("---")

# Upload do dataset
st.sidebar.header("📁 Upload do Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV",
    type=["csv"],
    help="Faça upload de um arquivo CSV para análise"
)

# Botão de limpar memória/cache
if st.sidebar.button("🗑️ Limpar Memória e Cache"):
    st.session_state.agent_memory = {
        "conclusions": [],
        "insights": [],
        "patterns_found": [],
        "analysis_history": [],
        "generated_plots": [],
        "llm_interactions": []
    }
    st.session_state.analysis_cache = {}
    st.sidebar.success("Memória e Cache limpos!")

# ==========================
# LÓGICA PRINCIPAL
# ==========================
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset carregado! {df.shape[0]:,} linhas × {df.shape[1]} colunas.")

        # Informações do dataset
        st.subheader("📋 Informações do Dataset")
        st.markdown(get_dataset_info(df))

        # Análise inicial do Gemini
        st.subheader("🧠 Análise Inicial do Gemini")
        initial_analysis = st.session_state.gemini_agent.analyze_dataset_initially(df)
        st.markdown(initial_analysis.get("full_response", "Análise inicial indisponível."))

        # Pergunta do usuário
        st.markdown("---")
        st.subheader("💬 Pergunte ao Agente")
        user_question = st.text_input("Digite sua pergunta aqui:")

        if user_question:
            with st.spinner("🔍 Gemini está analisando sua pergunta..."):
                response, visualization = st.session_state.gemini_agent.process_user_query(
                    user_question, df, st.session_state.agent_memory
                )
                st.markdown(response)

                if visualization and visualization.get("figure"):
                    st.pyplot(visualization["figure"])

        # Sugestões do Gemini
        st.markdown("---")
        st.subheader("💡 Sugestões Inteligentes")
        suggestions = st.session_state.gemini_agent.generate_smart_suggestions(df)
        for s in suggestions:
            st.markdown(f"- {s}")

        # Painel rápido de análises tradicionais
        st.markdown("---")
        st.subheader("⚡ Análises Tradicionais Rápidas")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Estatísticas Descritivas"):
                st.dataframe(perform_descriptive_analysis(df))
        with col2:
            if st.button("🔗 Correlações"):
                fig, conclusion = plot_correlation_heatmap(df)
                if fig:
                    st.pyplot(fig)
                    st.info(conclusion)
        with col3:
            if st.button("🎯 Clustering"):
                _, conclusion = perform_clustering_analysis(df)
                st.info(conclusion)

        # Gerar relatório PDF
        st.markdown("---")
        st.subheader("📄 Relatório PDF")
        if st.button("Gerar Relatório Completo"):
            with st.spinner("Gerando relatório..."):
                pdf_bytes = generate_pdf_report(df, llm_insights=initial_analysis.get("full_response", ""))
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="relatorio_analise.pdf">📥 Baixar Relatório PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Erro ao processar o dataset: {e}")

else:
    st.info("👆 Faça upload de um CSV para começar a análise.")


