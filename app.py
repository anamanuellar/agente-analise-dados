import streamlit as st
import pandas as pd
import numpy as np
import io
import base64

# Importações do novo GeminiAgent
from utils import (
    GeminiAgent,
    initialize_gemini_agent,
    get_dataset_info,
    generate_pdf_report,
    get_adaptive_suggestions
)

# Configuração da página
st.set_page_config(
    page_title="🤖 Agente Autônomo com Gemini",
    page_icon="🤖",
    layout="wide"
)

# === CONFIGURAÇÃO DO GEMINI ===
def setup_gemini():
    """Configura Google Gemini com fallback gracioso"""
    api_key = None
    
    # Tentar obter da configuração de secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        pass
    
    if not api_key:
        with st.sidebar:
            st.markdown("### 🔑 API Key do Gemini")
            api_key = st.text_input(
                "Cole sua API Key:",
                type="password",
                help="Obtenha gratuitamente em: https://aistudio.google.com/app/apikey"
            )
            
            if st.button("ℹ️ Como obter (GRATUITO)"):
                st.info("""
                **Completamente GRATUITO:**
                1. Acesse: https://aistudio.google.com/app/apikey
                2. Login com conta Google
                3. Clique "Create API Key"
                4. Cole aqui
                
                ✅ Sem cartão de crédito
                ✅ Uso generoso gratuito
                """)
    
    return api_key is not None, api_key

# Inicializar sistema Gemini
gemini_available, gemini_key = setup_gemini()

# Inicializar agente Gemini
if 'gemini_agent' not in st.session_state:
    st.session_state.gemini_agent = initialize_gemini_agent()

# Configurar Gemini se disponível
if gemini_available and gemini_key and st.session_state.gemini_agent:
    try:
        st.session_state.gemini_agent.configure_gemini(gemini_key)
        gemini_configured = True
    except Exception as e:
        st.error(f"Erro ao configurar Gemini: {e}")
        gemini_configured = False
else:
    gemini_configured = False

# Inicializar memória Gemini
if 'gemini_memory' not in st.session_state:
    st.session_state.gemini_memory = []

# === INTERFACE PRINCIPAL ===

st.title("🤖 Agente Autônomo de Análise de Dados")
st.markdown("*Powered by Google Gemini - IA Generativa para Análise Inteligente*")

# Status do sistema
if gemini_configured:
    st.success("🧠 **Gemini Configurado:** IA Generativa Ativa para Análise Inteligente")
else:
    st.warning("⚠️ **Configure Gemini na barra lateral para IA completa**")

st.markdown("---")

# === SIDEBAR ===

st.sidebar.header("📁 Upload do Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV",
    type=['csv'],
    help="Upload do arquivo para análise com IA"
)

# Sidebar - Status do Gemini
st.sidebar.markdown("---")
st.sidebar.header("🧠 Status da IA")
if gemini_configured:
    st.sidebar.success("✅ Gemini Ativo")
    st.sidebar.write(f"**Modelo:** {st.session_state.gemini_agent.model_name}")
else:
    st.sidebar.error("❌ Gemini não configurado")

# Sidebar - Memória do Agente
st.sidebar.markdown("---")
st.sidebar.header("🗃️ Memória do Agente")
if st.session_state.gemini_memory:
    st.sidebar.write(f"**Análises realizadas:** {len(st.session_state.gemini_memory)}")
    if st.sidebar.button("📋 Ver Memória Completa"):
        if gemini_configured:
            memory_summary = st.session_state.gemini_agent.get_full_memory_summary()
            st.sidebar.text_area("Memória:", memory_summary, height=300)
        else:
            st.sidebar.json(st.session_state.gemini_memory)
else:
    st.sidebar.write("*Aguardando primeira análise...*")

if st.sidebar.button("🗑️ Limpar Memória"):
    st.session_state.gemini_memory = []
    st.sidebar.success("Memória limpa!")

# === LÓGICA PRINCIPAL ===

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # === PRÉVIA DO DATASET ===
        st.markdown("---")
        st.subheader("👀 Prévia do Dataset")

        with col2:
            st.write(f"• **Arquivo:** {uploaded_file.name}")
            st.write(f"• **Upload em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
                
        with col1:
            st.write("**Primeiras 5 linhas:**")
            st.dataframe(df.head(), use_container_width=True)
        
        # === INFORMAÇÕES DETALHADAS ===
        with st.expander("📋 Ver Informações Detalhadas"):
            dataset_info_text = get_dataset_info(df)
            st.markdown(dataset_info_text)

        # === ANÁLISE INICIAL COM GEMINI ===
        if gemini_configured and 'initial_analysis_done' not in st.session_state:
            with st.spinner("🧠 IA analisando dataset..."):
                initial_analysis = st.session_state.gemini_agent.analyze_dataset_initially(df)
                st.session_state.initial_analysis = initial_analysis
                st.session_state.initial_analysis_done = True
        
        # Mostrar análise inicial se disponível
        if 'initial_analysis' in st.session_state and gemini_configured:
            st.subheader("🧠 Análise Inicial da IA")
            analysis = st.session_state.initial_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Identificação do Domínio:**")
                st.info(analysis.get('dataset_type', 'Não identificado'))
                
                st.write("**📊 Características Principais:**")
                for char in analysis.get('key_characteristics', []):
                    st.write(f"• {char}")
                    
            with col2:
                st.write("**💡 Análises Recomendadas:**")
                for rec in analysis.get('recommended_analyses', []):
                    st.write(f"• {rec}")
                
                st.write("**🔮 Insights Potenciais:**")
                for insight in analysis.get('potential_insights', []):
                    st.write(f"• {insight}")

            # Mostrar resposta completa em expander
            with st.expander("📄 Ver Análise Completa da IA"):
                st.markdown(analysis.get('full_response', 'Análise não disponível'))

        # === CHAT COM IA ===
        st.markdown("---")
        st.subheader("💬 Converse com a IA sobre seus Dados")
        
        # Gerenciar estado da pergunta usando session_state
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # Campo de pergunta
        user_question = st.text_input(
            "Faça uma pergunta sobre seus dados:",
            value=st.session_state.current_question,
            placeholder="Ex: Quais são as correlações mais importantes?",
            key="user_question_input"
        )
        
        # Atualizar a pergunta atual
        if user_question != st.session_state.current_question:
            st.session_state.current_question = user_question

        # === SUGESTÕES INTELIGENTES ===
        st.markdown("**💡 Sugestões da IA:**")
        
        if gemini_configured and 'initial_analysis' in st.session_state:
            # Usar sugestões inteligentes do Gemini
            if 'smart_suggestions' not in st.session_state:
                with st.spinner("Gerando sugestões inteligentes..."):
                    st.session_state.smart_suggestions = st.session_state.gemini_agent.generate_smart_suggestions(df)
            
            # Criar botões clicáveis para as sugestões
            cols = st.columns(2)
            for i, suggestion in enumerate(st.session_state.smart_suggestions):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(f"🔍 {suggestion[:50]}...", key=f"suggestion_{i}", use_container_width=True):
                        st.session_state.current_question = suggestion
                        st.rerun()
        else:
            # Usar sugestões básicas
            basic_suggestions = get_adaptive_suggestions(df)
            for suggestion in basic_suggestions:
                st.markdown(f"• {suggestion}")
        
        st.markdown("---")

        # === PROCESSAMENTO DE PERGUNTA ===
        # Usar a pergunta atual do session_state
        question_to_process = st.session_state.current_question
        
        if question_to_process:
            if gemini_configured:
                # MODO IA: Processamento inteligente completo
                with st.spinner("🤖 IA processando sua pergunta..."):
                    try:
                        response, visualization = st.session_state.gemini_agent.process_user_query(question_to_process, df)
                        
                        # Mostrar resposta da IA
                        st.subheader("🧠 Resposta da IA")
                        st.markdown(response)
                        
                        # Mostrar visualização se houver
                        if visualization and visualization.get("status") == "success":
                            st.subheader("📊 Visualização Gerada")
                            fig = visualization.get("figure")
                            if fig:
                                st.pyplot(fig)
                            else:
                                st.error("Erro ao exibir gráfico")
                        elif visualization and visualization.get("status") == "error":
                            st.error(f"Erro na visualização: {visualization.get('error_message')}")
                            with st.expander("🔧 Debug - Código gerado"):
                                st.code(visualization.get("code_executed", "Código não disponível"), language="python")
                        
                        # Limpar a pergunta após processar
                        if st.button("✅ Nova Pergunta"):
                            st.session_state.current_question = ""
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Erro ao processar pergunta: {str(e)}")
                        
            else:
                # MODO BÁSICO: Sem IA configurada
                st.warning("Configure Gemini na barra lateral para análise inteligente")
                st.info("Sua pergunta foi registrada. Configure a IA para obter resposta inteligente.")

        # === PAINEL DE ANÁLISES RÁPIDAS ===
        st.markdown("---")
        st.subheader("⚡ Análises Rápidas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Estatísticas Descritivas", use_container_width=True):
                if gemini_configured:
                    response, viz = st.session_state.gemini_agent.process_user_query(
                        "Mostre estatísticas descritivas completas do dataset", df
                    )
                    st.markdown(response)
                else:
                    st.dataframe(df.describe())
        
        with col2:
            if st.button("🔗 Análise de Correlações", use_container_width=True):
                if gemini_configured:
                    response, viz = st.session_state.gemini_agent.process_user_query(
                        "Analise as correlações entre todas as variáveis numéricas e crie um heatmap", df
                    )
                    st.markdown(response)
                    if viz and viz.get("status") == "success":
                        st.pyplot(viz.get("figure"))
                else:
                    st.info("Configure Gemini para análise inteligente")
        
        with col3:
            if st.button("🎯 Clustering Automático", use_container_width=True):
                if gemini_configured:
                    response, viz = st.session_state.gemini_agent.process_user_query(
                        "Faça uma análise de clustering automática dos dados", df
                    )
                    st.markdown(response)
                    if viz and viz.get("status") == "success":
                        st.pyplot(viz.get("figure"))
                else:
                    st.info("Configure Gemini para análise inteligente")

        # === CENTRAL DE RELATÓRIOS ===
        st.markdown("---")
        st.subheader("📊 Central de Relatórios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Resumo Completo da IA", type="primary", use_container_width=True):
                if gemini_configured and st.session_state.gemini_memory:
                    summary = st.session_state.gemini_agent.get_full_memory_summary()
                    st.markdown(summary)
                else:
                    st.warning("Execute algumas análises primeiro ou configure Gemini")
        
        with col2:
            if st.button("📄 Relatório PDF", type="secondary", use_container_width=True):
                if gemini_configured and st.session_state.gemini_memory:
                    with st.spinner('Gerando relatório PDF...'):
                        try:
                            pdf_content = generate_pdf_report(df, st.session_state.gemini_agent)
                            
                            b64_pdf = base64.b64encode(pdf_content).decode()
                            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="relatorio_gemini_agent.pdf">📥 Download Relatório</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            st.success("✅ Relatório gerado com sucesso!")
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
                else:
                    st.warning("⚠️ Execute algumas análises primeiro")

        # === HISTÓRICO DE CONVERSAS ===
        if gemini_configured and st.session_state.gemini_agent.conversation_history:
            st.markdown("---")
            st.subheader("💬 Histórico de Conversas")
            
            for i, conv in enumerate(st.session_state.gemini_agent.conversation_history[-5:]):  # Últimas 5
                with st.expander(f"Conversa {i+1}: {conv.get('content', 'N/A')[:50]}..."):
                    st.write(f"**Tipo:** {conv['role']}")
                    st.write(f"**Conteúdo:** {conv['content']}")
                    st.write(f"**Timestamp:** {conv['timestamp']}")
    
    except Exception as e:
        st.error(f"❌ Erro ao processar o dataset: {str(e)}")
        st.code(str(e))

else:
    # === PÁGINA INICIAL ===
    st.info("👆 **Faça upload de um arquivo CSV na barra lateral para começar.**")
    
    st.markdown("""
    ## 🚀 **Agente Autônomo com IA Generativa**
    
    ### 🧠 **Powered by Google Gemini**
    
    #### **✨ Capacidades Avançadas:**
    - 🔍 **Análise Inicial Automática:** IA identifica automaticamente o tipo e características dos dados
    - 💬 **Chat Inteligente:** Converse em linguagem natural sobre seus dados
    - 📊 **Visualizações Automáticas:** IA gera gráficos relevantes automaticamente
    - 🎯 **Sugestões Contextuais:** Perguntas inteligentes baseadas nos seus dados
    - 📋 **Relatórios Executivos:** Resumos profissionais gerados pela IA
    - 🧠 **Memória Persistente:** IA lembra de todas as análises realizadas
    
    #### **🎯 Exemplos de Perguntas:**
    - *"Quais são os principais insights sobre este dataset?"*
    - *"Mostre correlações importantes e crie um heatmap"*
    - *"Detecte outliers e explique o que encontrou"*
    - *"Faça clustering e visualize os grupos"*
    - *"Analise padrões temporais nos dados"*
    
    #### **🔑 Como Usar:**
    
    1. **Configure Gemini (Gratuito):**
       - Obtenha API Key em: https://aistudio.google.com/app/apikey
       - Cole na barra lateral
       - Sem cartão de crédito necessário
    
    2. **Carregue seus Dados:**
       - Upload do CSV
       - IA analisa automaticamente
       - Recebe insights imediatos
    
    3. **Converse com a IA:**
       - Faça perguntas em português
       - IA gera visualizações
       - Obtém insights profissionais
    
    ---
    
    🧠 **Powered by Google Gemini** | 🎯 **IA que Entende Dados** | 🚀 **Análise Inteligente**
    """)

# === RODAPÉ ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
🤖 <strong>Agente Autônomo de Análise de Dados</strong><br>
🧠 Powered by Google Gemini - IA Generativa<br>
Desenvolvido para análise inteligente de dados | 2025<br>
</div>
""", unsafe_allow_html=True)


