import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
import google.generativeai as genai 

# Importar a classe GeminiAgent e outras funções auxiliares do utils.py
from utils import GeminiAgent, get_dataset_info, generate_pdf_report

# --- CONFIGURAÇÃO INICIAL --- 

st.set_page_config(
    page_title="Agente de Análise de Dados com Gemini",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo de Análise de Dados com Gemini")
st.markdown("---")

# Inicializar o GeminiAgent no session_state
if "gemini_agent" not in st.session_state:
    st.session_state.gemini_agent = GeminiAgent()

# --- SIDEBAR: CONFIGURAÇÃO E UPLOAD ---
st.sidebar.header("⚙️ Configurações")

# Input para a API Key do Gemini
api_key_input = st.sidebar.text_input(
    "Sua API Key do Google Gemini",
    type="password",
    value=os.getenv("GEMINI_API_KEY", "") 
)

if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input
    st.session_state.gemini_agent.configure_gemini(api_key_input)
    st.sidebar.success("API Key configurada!")
else:
    st.sidebar.warning("Por favor, insira sua API Key do Google Gemini para usar o agente.")
    st.stop()

st.sidebar.markdown("--- ")
st.sidebar.header("📁 Upload do Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV",
    type=["csv"],
    help="Faça upload de um arquivo CSV para análise"
)

# --- LÓGICA PRINCIPAL --- 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df # Armazenar o DataFrame na sessão

    st.success("Dataset carregado com sucesso!")
    st.subheader("📊 Pré-visualização dos Dados")
    st.dataframe(df.head())

    # Análise inicial do dataset pelo GeminiAgent
    if "initial_analysis_done" not in st.session_state or not st.session_state.initial_analysis_done:
        with st.spinner("🤖 Agente analisando o dataset inicialmente..."):
            initial_analysis_results = st.session_state.gemini_agent.analyze_dataset_initially(df)
            st.session_state.initial_analysis_results = initial_analysis_results
            st.session_state.initial_analysis_done = True

    st.subheader("🧠 Análise Inicial do Agente")
    st.markdown(st.session_state.initial_analysis_results["full_response"])

    st.markdown("---")
    st.subheader("💬 Converse com o Agente")
    user_query = st.text_input("Faça sua pergunta sobre os dados:", key="user_query_input")

    # Sugestões adaptativas baseadas na análise inicial
    st.markdown("**💡 Sugestões de Perguntas:**")
    cols = st.columns(3)
    # As sugestões agora vêm do GeminiAgent
    if "smart_suggestions" not in st.session_state:
        st.session_state.smart_suggestions = st.session_state.gemini_agent.generate_smart_suggestions(df)
    
    for i, suggestion in enumerate(st.session_state.smart_suggestions):
        with cols[i % 3]:
            st.info(suggestion)

    if user_query:
        with st.spinner("🤖 Agente processando sua pergunta..."):
            response_text, visualization_data = st.session_state.gemini_agent.process_user_query(
                user_query, df
            )
            st.write(response_text)
            if visualization_data and visualization_data["status"] == "success" and visualization_data["figure"]:
                st.pyplot(visualization_data["figure"])
            elif visualization_data and visualization_data["status"] == "error":
                st.error(f"Erro ao gerar visualização: {visualization_data["error_message"]}")

    st.markdown("---")
    st.subheader("⚡ Central de Inteligência do Agente")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Executar Resumo Das Análises", use_container_width=True):
            with st.spinner("Gerando resumo completo das análises..."):
                summary = st.session_state.gemini_agent.get_full_memory_summary()
                st.session_state.summary_display = summary

    with col2:
        if st.button("📄 Gerar Relatório PDF Completo", use_container_width=True):
            with st.spinner("Gerando relatório PDF..."):
                pdf_bytes = generate_pdf_report(df, st.session_state.gemini_agent)
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="Agentes_Autonomos_Relatorio_da_Atividade_Extra.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    if "summary_display" in st.session_state and st.session_state.summary_display:
        st.markdown(st.session_state.summary_display)

else:
    st.info("👆 Faça upload de um arquivo CSV na barra lateral para começar a análise.")
    st.markdown("""
    ### 🚀 Como usar:
    1. Faça upload de um arquivo CSV na barra lateral.
    2. Insira sua API Key do Google Gemini na sidebar.
    3. O agente fará uma análise inicial e sugerirá perguntas.
    4. Faça suas perguntas em linguagem natural na caixa de texto.
    5. Use a Central de Inteligência para resumos e relatórios.
    """)

st.markdown("---")
st.markdown("""
<div style=\'text-align: center; color: #666; font-size: 14px;\'>
🤖 <strong>Agente Autônomo de Análise de Dados com Gemini</strong>  

Desenvolvido por Ana Manuella - <strong>I2A2 Academy</strong> | Setembro 2025  

</div>
""", unsafe_allow_html=True)
