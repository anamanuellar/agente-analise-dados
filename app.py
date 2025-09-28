import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import google.generativeai as genai

from utils import (
    HybridGeminiAgent,
    add_to_memory,
    get_memory_summary,
    get_dataset_info,
    generate_complete_analysis_summary,
    analyze_frequent_values,
    perform_descriptive_analysis,
    plot_distribution,
    plot_correlation_heatmap,
    analyze_temporal_patterns,
    perform_clustering_analysis,
    detect_outliers,
    analyze_balance,
    generate_pdf_report,
    interpret_question,
    get_adaptive_suggestions
)

if 'hybrid_agent' not in st.session_state:
    agent = initialize_hybrid_agent()
    if agent is None:
        st.stop()  # Para a execução se não conseguir configurar
    st.session_state.hybrid_agent = agent
else:
    agent = st.session_state.hybrid_agent

# Configuração da página
st.set_page_config(
    page_title="🤖 Agente Híbrido - Gemini + Funções Robustas",
    page_icon="🤖",
    layout="wide"
)

# === CONFIGURAÇÃO DO GEMINI ===
def setup_gemini():
    """Configura Google Gemini com fallback gracioso"""
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    
    if not api_key:
        with st.sidebar:
            st.markdown("### 🔑 API Key do Gemini")
            api_key = st.text_input(
                "Cole sua API Key:",
                type="password",
                help="Obtenha gratuitamente em: https://makersuite.google.com/app/apikey"
            )
            
            if st.button("ℹ️ Como obter (GRATUITO)"):
                st.info("""
                **Completamente GRATUITO:**
                1. Acesse: https://makersuite.google.com/app/apikey
                2. Login com conta Google
                3. Clique "Create API Key"
                4. Cole aqui
                
                ✅ Sem cartão de crédito
                ✅ Uso generoso gratuito
                """)
    
    return api_key is not None, api_key

# Inicializar sistema híbrido
gemini_available, gemini_key = setup_gemini()

if 'hybrid_agent' not in st.session_state:
    st.session_state.hybrid_agent = HybridGeminiAgent()
    if gemini_available and gemini_key:
        st.session_state.hybrid_agent.configure_gemini(gemini_key)

# Inicializar memória do agente
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = {
        'conclusions': [],
        'insights': [],
        'patterns_found': [],
        'analysis_history': [],
        'generated_plots': []
    }

# Cache para análises
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Configurações avançadas
if 'max_sample_size' not in st.session_state:
    st.session_state.max_sample_size = 5000
if 'contamination_rate' not in st.session_state:
    st.session_state.contamination_rate = 0.10

# === INTERFACE PRINCIPAL ===

st.title("🤖 Agente Híbrido: IA + Análises Robustas")
st.markdown("*Interface completa da v1 + Inteligência do Gemini da v2*")

# Status do sistema
if gemini_available:
    st.success("🧠 **Sistema Híbrido Ativo:** Gemini (IA) + Funções Robustas")
else:
    st.warning("⚠️ **Modo Básico:** Configure Gemini na barra lateral para IA completa")

st.markdown("---")

# === SIDEBAR ===

st.sidebar.header("📁 Upload do Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV",
    type=['csv'],
    help="Upload do arquivo para análise híbrida"
)

# Sidebar - Memória do Agente
st.sidebar.markdown("---")
st.sidebar.header("🧠 Memória do Agente")
if st.session_state.agent_memory['conclusions']:
    st.sidebar.write(f"**Conclusões:** {len(st.session_state.agent_memory['conclusions'])}")
    st.sidebar.write(f"**Análises:** {len(set(st.session_state.agent_memory['analysis_history']))}")
    if st.sidebar.button("📋 Ver Memória Completa"):
        st.sidebar.json(st.session_state.agent_memory)
else:
    st.sidebar.write("*Aguardando primeira análise...*")

if st.sidebar.button("🗑️ Limpar Memória e Cache"):
    st.session_state.agent_memory = {
        'conclusions': [],
        'insights': [],
        'patterns_found': [],
        'analysis_history': [],
        'generated_plots': []
    }
    st.session_state.analysis_cache = {}
    st.sidebar.success("Memória e Cache limpos!")

# Sidebar - Configurações
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações")
st.session_state.max_sample_size = st.sidebar.slider(
    "Tamanho máximo da amostra:", 1000, 20000, st.session_state.max_sample_size
)
st.session_state.contamination_rate = st.sidebar.slider(
    "Taxa de contaminação para outliers:", 0.01, 0.20, st.session_state.contamination_rate
)

# === LÓGICA PRINCIPAL ===

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    agent = st.session_state.hybrid_agent
    
    # Análise inicial
    with st.spinner("Analisando dataset..."):
        initial_analysis = agent.analyze_dataset_initially(df)
        st.write("Dataset analisado com sucesso!")

        # === ANÁLISE INICIAL COM GEMINI (SE DISPONÍVEL) ===
        if gemini_available and 'initial_analysis_done' not in st.session_state:
            with st.spinner("🧠 Agente híbrido analisando dataset com Gemini..."):
                initial_analysis = st.session_state.hybrid_agent.analyze_dataset_initially(df)
                st.session_state.initial_analysis = initial_analysis
                st.session_state.initial_analysis_done = True
        
        # Mostrar análise inicial se disponível
        if 'initial_analysis' in st.session_state and gemini_available:
            st.subheader("🧠 Análise Inicial da IA")
            analysis = st.session_state.initial_analysis
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Tipo de Dataset:**")
                st.info(analysis.get('dataset_type', 'Não identificado'))
                
                st.write("**📊 Características Principais:**")
                for char in analysis.get('key_characteristics', []):
                    st.write(f"• {char}")
                    
            with col2:
                st.write("**💡 Análises Recomendadas:**")
                for rec in analysis.get('recommended_analyses', []):
                    st.write(f"• {rec}")
                
                # Estatísticas básicas
                st.write("**📈 Estatísticas:**")
                st.write(f"• Linhas: {analysis.get('shape', [0, 0])[0]:,}")
                st.write(f"• Colunas: {analysis.get('shape', [0, 0])[1]}")
                st.write(f"• Completude: {analysis.get('completeness', 0):.1f}%")

        # === INFORMAÇÕES DO DATASET ===
        st.markdown("---")
        st.subheader("📋 Informações Gerais do Dataset")
        dataset_info_text = get_dataset_info(df)
        st.markdown(dataset_info_text)

        # === CAMPO DE PERGUNTA COM IA ===
        st.markdown("---")
        st.subheader("💬 Faça uma Pergunta ao Agente Híbrido")
        
        user_question = st.text_input("Digite sua pergunta aqui:", key="user_question")

        # === SUGESTÕES ADAPTATIVAS (HÍBRIDAS) ===
        st.markdown("**💡 Sugestões de Perguntas:**")
        
        if gemini_available and 'initial_analysis' in st.session_state:
            # Usar sugestões inteligentes do Gemini
            if 'smart_suggestions' not in st.session_state:
                with st.spinner("Gerando sugestões inteligentes..."):
                    st.session_state.smart_suggestions = st.session_state.hybrid_agent.generate_smart_suggestions(df)
            
            for suggestion in st.session_state.smart_suggestions:
                st.markdown(f"• {suggestion}")
        else:
            # Usar sugestões básicas
            suggestions = get_adaptive_suggestions(df)
            for suggestion in suggestions:
                st.markdown(suggestion)
        
        st.markdown("---")

        # === PROCESSAMENTO DE PERGUNTA HÍBRIDO ===
        if user_question:
            if gemini_available:
                # MODO HÍBRIDO: IA + Funções Robustas
                with st.spinner("🤖 Agente híbrido processando com IA..."):
                    # Interpretação inteligente
                    query_interpretation = st.session_state.hybrid_agent.interpret_query_intelligently(
                        user_question, df
                    )
                    
                    st.write(f"🔍 **IA interpretou como:** {query_interpretation['category'].replace('_', ' ').title()}")
                    if query_interpretation['specific_columns']:
                        st.write(f"🎯 **Colunas identificadas:** {', '.join(query_interpretation['specific_columns'])}")
            else:
                # MODO BÁSICO: Apenas regras
                query_interpretation = {
                    'category': interpret_question(user_question, df),
                    'specific_columns': [],
                    'confidence': 'medium'
                }
                st.write(f"🔍 **Pergunta interpretada como:** {query_interpretation['category'].replace('_', ' ').title()}")

            analysis_type = query_interpretation['category']
            specific_columns = query_interpretation['specific_columns']

            # === EXECUÇÃO DAS ANÁLISES (FUNÇÕES ROBUSTAS) ===
            
            if analysis_type == 'distribution':
                st.subheader("📈 Análise de Distribuição")
                
                if specific_columns and specific_columns[0] in df.select_dtypes(include=[np.number]).columns:
                    col_to_analyze = specific_columns[0]
                else:
                    col_to_analyze = st.selectbox("Selecione a coluna:", 
                                                 df.select_dtypes(include=[np.number]).columns)
                
                if col_to_analyze:
                    with st.spinner(f'Gerando distribuição para {col_to_analyze}...'):
                        fig, conclusion = plot_distribution(df, col_to_analyze)
                        st.pyplot(fig)
                        
                        # Resposta híbrida com IA
                        if gemini_available:
                            ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                query_interpretation, conclusion, df
                            )
                            st.info(f"🧠 **Insights da IA:** {ai_response}")
                        else:
                            st.info(conclusion)
            
            elif analysis_type == 'correlation':
                st.subheader("🔗 Análise de Correlação")
                with st.spinner('Calculando correlações...'):
                    fig, conclusion = plot_correlation_heatmap(df)
                    if fig:
                        st.pyplot(fig)
                        
                        # Resposta híbrida
                        if gemini_available:
                            ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                query_interpretation, conclusion, df
                            )
                            st.info(f"🧠 **Insights da IA:** {ai_response}")
                        else:
                            st.info(conclusion)
                    else:
                        st.error("❌ Necessário pelo menos 2 colunas numéricas")
            
            elif analysis_type == 'temporal':
                st.subheader("⏰ Análise de Padrões Temporais")
                time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                
                if time_cols:
                    if specific_columns and specific_columns[0] in time_cols:
                        selected_time_col = specific_columns[0]
                    else:
                        selected_time_col = st.selectbox("Selecione a coluna temporal:", time_cols)
                    
                    if selected_time_col:
                        with st.spinner(f'Analisando padrões temporais...'):
                            fig, conclusion = analyze_temporal_patterns(df, selected_time_col)
                            if fig:
                                st.pyplot(fig)
                            
                            # Resposta híbrida
                            if gemini_available:
                                ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                    query_interpretation, conclusion, df
                                )
                                st.info(f"🧠 **Insights da IA:** {ai_response}")
                            else:
                                st.info(conclusion)
                else:
                    st.warning("❌ Nenhuma coluna temporal encontrada no dataset")
            
            elif analysis_type == 'clustering':
                st.subheader("🎯 Análise de Clustering")
                if st.button("🎯 Executar Clustering Inteligente"):
                    with st.spinner('Executando clustering...'):
                        fig, conclusion = perform_clustering_analysis(
                            df, sample_size=st.session_state.max_sample_size
                        )
                        
                        # Resposta híbrida
                        if gemini_available:
                            ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                query_interpretation, conclusion, df
                            )
                            st.info(f"🧠 **Insights da IA:** {ai_response}")
                        else:
                            st.info(conclusion)
            
            elif analysis_type == 'frequency':
                st.subheader("📊 Análise de Valores Frequentes")
                if st.button("🔍 Analisar Frequências"):
                    with st.spinner('Analisando frequências...'):
                        freq_results = analyze_frequent_values(df)
                        
                        if freq_results:
                            for col, data in freq_results.items():
                                st.write(f"**Coluna: {col}** ({data['type']})")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("*Mais frequentes:*")
                                    for val, count in data['most_frequent'].items():
                                        st.write(f"• {val}: {count}")
                                
                                with col2:
                                    st.write("*Menos frequentes:*")
                                    for val, count in data['least_frequent'].items():
                                        st.write(f"• {val}: {count}")
                                
                                st.write(f"Únicos: {data['unique_count']}, Nulos: {data['null_count']}")
                                st.markdown("---")
                            
                            # Resposta híbrida
                            if gemini_available:
                                ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                    query_interpretation, f"Análise de frequência de {len(freq_results)} colunas", df
                                )
                                st.info(f"🧠 **Insights da IA:** {ai_response}")
                        else:
                            st.info("Nenhuma coluna categórica encontrada")
            
            elif analysis_type == 'memory':
                st.subheader("🧠 Memória do Agente")
                memory_summary = get_memory_summary()
                st.markdown(memory_summary)
                
                if st.session_state.agent_memory['conclusions']:
                    st.write("**📚 Histórico Detalhado:**")
                    for i, entry in enumerate(st.session_state.agent_memory['conclusions'], 1):
                        with st.expander(f"Análise {i}: {entry['analysis_type'].replace('_', ' ').title()}"):
                            st.write(f"**🔍 Conclusão:** {entry['conclusion']}")
                            st.write(f"**⏰ Timestamp:** {entry['timestamp']}")
            
            elif analysis_type == 'descriptive':
                st.subheader("📈 Análise Descritiva")
                if st.button("📊 Gerar Estatísticas Descritivas"):
                    with st.spinner('Gerando análise descritiva...'):
                        desc_stats = perform_descriptive_analysis(df)
                        st.dataframe(desc_stats, use_container_width=True)
                        
                        # Informações detalhadas
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        info_str = buffer.getvalue()
                        st.text(info_str)
            
            elif analysis_type == 'outliers':
                st.subheader("🔍 Detecção de Outliers")
                
                if specific_columns and specific_columns[0] in df.select_dtypes(include=[np.number]).columns:
                    col_to_analyze = specific_columns[0]
                else:
                    col_to_analyze = st.selectbox("Selecione a coluna:", 
                                                 df.select_dtypes(include=[np.number]).columns)
                
                if col_to_analyze and st.button(f"Detectar Outliers em {col_to_analyze}"):
                    with st.spinner(f'Detectando outliers...'):
                        fig, conclusion = detect_outliers(df, col_to_analyze)
                        
                        # Resposta híbrida
                        if gemini_available:
                            ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                query_interpretation, conclusion, df
                            )
                            st.info(f"🧠 **Insights da IA:** {ai_response}")
                        else:
                            st.info(conclusion)
            
            elif analysis_type == 'balance':
                st.subheader("⚖️ Análise de Balanceamento")
                binary_cols = [col for col in df.columns if df[col].nunique() == 2]
                
                if binary_cols:
                    if specific_columns and specific_columns[0] in binary_cols:
                        selected_col = specific_columns[0]
                    else:
                        selected_col = st.selectbox("Selecione a coluna binária:", binary_cols)
                    
                    if selected_col and st.button(f"Analisar Balanceamento"):
                        with st.spinner('Analisando balanceamento...'):
                            fig, conclusion = analyze_balance(df, selected_col)
                            if fig:
                                st.pyplot(fig)
                            
                            # Resposta híbrida
                            if gemini_available:
                                ai_response = st.session_state.hybrid_agent.generate_intelligent_response(
                                    query_interpretation, conclusion, df
                                )
                                st.info(f"🧠 **Insights da IA:** {ai_response}")
                            else:
                                st.info(conclusion)
                else:
                    st.warning("❌ Nenhuma coluna binária encontrada")
            
            elif analysis_type == 'insights':
                st.subheader("🎯 Conclusões e Insights")
                if gemini_available:
                    with st.spinner('🤖 IA gerando conclusões executivas...'):
                        conclusions = st.session_state.hybrid_agent.generate_executive_conclusions(
                            df, st.session_state.agent_memory
                        )
                        st.markdown(conclusions)
                else:
                    memory_summary = get_memory_summary()
                    st.markdown(memory_summary)
                    st.info("Configure Gemini para conclusões inteligentes")
            
            else:  # general
                st.info("Não entendi sua pergunta. Tente usar as sugestões ou seja mais específico.")
        
        # === PAINEL DE ANÁLISES RÁPIDAS (VERSÃO 1 MANTIDA) ===
        st.markdown("---")
        st.subheader("⚡ Painel de Análises Rápidas")
        
        col_buttons, col_results = st.columns([1, 2])
        
        with col_buttons:
            st.write("**Análises Rápidas:**")
            
            if st.button("📊 Estatísticas Descritivas", use_container_width=True):
                st.session_state.quick_analysis = "descriptive"
            
            if st.button("🔗 Mapa de Correlação", use_container_width=True):
                st.session_state.quick_analysis = "correlation"
            
            if st.button("⏰ Padrões Temporais", use_container_width=True):
                st.session_state.quick_analysis = "temporal"
            
            if st.button("🎯 Clustering Automático", use_container_width=True):
                st.session_state.quick_analysis = "clustering"
            
            if st.button("📊 Valores Frequentes", use_container_width=True):
                st.session_state.quick_analysis = "frequency"
            
            if st.button("🧠 Memória do Agente", use_container_width=True):
                st.session_state.quick_analysis = "memory"
        
        with col_results:
            if 'quick_analysis' in st.session_state:
                analysis_type = st.session_state.quick_analysis
                
                if analysis_type == "descriptive":
                    with st.spinner('Calculando estatísticas...'):
                        desc_stats = perform_descriptive_analysis(df)
                        st.dataframe(desc_stats, use_container_width=True)
                
                elif analysis_type == "correlation":
                    with st.spinner('Calculando correlações...'):
                        fig, conclusion = plot_correlation_heatmap(df)
                        if fig:
                            st.pyplot(fig)
                            st.info(conclusion)
                        else:
                            st.error("Necessário pelo menos 2 colunas numéricas")
                
                elif analysis_type == "temporal":
                    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if time_cols:
                        with st.spinner('Analisando padrões temporais...'):
                            fig, conclusion = analyze_temporal_patterns(df, time_cols[0])
                            if fig:
                                st.pyplot(fig)
                            st.info(conclusion)
                    else:
                        st.warning("Coluna temporal não encontrada")
                
                elif analysis_type == "clustering":
                    with st.spinner('Executando clustering...'):
                        fig, conclusion = perform_clustering_analysis(
                            df, sample_size=st.session_state.max_sample_size
                        )
                        st.info(conclusion)
                
                elif analysis_type == "frequency":
                    with st.spinner('Analisando frequências...'):
                        freq_results = analyze_frequent_values(df)
                        if freq_results:
                            for col, data in list(freq_results.items())[:3]:
                                st.write(f"**{col}:** Mais frequente = {list(data['most_frequent'].keys())[0]}")
                        else:
                            st.info("Nenhuma coluna categórica encontrada")
                
                elif analysis_type == "memory":
                    memory_summary = get_memory_summary()
                    st.markdown(memory_summary)
            
            else:
                st.info("👈 Selecione uma análise rápida para ver os resultados")
        
        # === CENTRAL DE INTELIGÊNCIA ===
        st.markdown("---")
        st.subheader("🤖 Central de Inteligência Híbrida")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Resumo Completo das Análises", type="primary", use_container_width=True):
                with st.spinner('Gerando resumo completo...'):
                    complete_summary = generate_complete_analysis_summary(df)
                    st.markdown(complete_summary)
        
        with col2:
            if st.button("📄 Relatório PDF Híbrido", type="secondary", use_container_width=True):
                if st.session_state.agent_memory['conclusions'] or gemini_available:
                    with st.spinner('Gerando relatório híbrido...'):
                        try:
                            # Incluir insights do Gemini se disponível
                            gemini_insights = None
                            if gemini_available and 'initial_analysis' in st.session_state:
                                gemini_insights = st.session_state.initial_analysis.get('llm_analysis', '')
                            
                            pdf_content = generate_pdf_report(df, gemini_insights)
                            
                            b64_pdf = base64.b64encode(pdf_content).decode()
                            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="relatorio_hibrido_gemini.pdf">📥 Download Relatório Híbrido</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            st.success("✅ Relatório híbrido gerado com sucesso!")
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
                else:
                    st.warning("⚠️ Execute algumas análises primeiro")
        
        # === CONCLUSÕES EXECUTIVAS COM IA ===
        if gemini_available:
            st.markdown("---")
            st.subheader("🎓 Conclusões Executivas da IA")
            
            if st.button("🧠 Gerar Conclusões Inteligentes", type="primary", use_container_width=True):
                with st.spinner('🤖 IA gerando conclusões executivas...'):
                    executive_conclusions = st.session_state.hybrid_agent.generate_executive_conclusions(
                        df, st.session_state.agent_memory
                    )
                    st.markdown(executive_conclusions)
    
    except Exception as e:
        st.error(f"❌ Erro ao processar o dataset: {str(e)}")
        st.code(str(e))

else:
    # === PÁGINA INICIAL ===
    st.info("👆 **Faça upload de um arquivo CSV na barra lateral para começar.**")
    
    st.markdown("""
    ## 🚀 **Agente Híbrido: O Melhor dos Dois Mundos**
    
    ### 🧠 **Sistema Híbrido Inovador:**
    
    #### **✨ Combinação Perfeita:**
    - 🧠 **Gemini (Google):** Interpretação inteligente e insights contextuais
    - 🔧 **Funções Robustas:** Análises confiáveis que sempre funcionam
    - 🎨 **Interface Completa:** Todos os recursos da versão 1
    - 🚀 **Inteligência Adaptativa:** IA que entende seus dados
    
    #### **🎯 Funcionalidades Híbridas:**
    - ✅ **Análise Inicial com IA:** Gemini identifica automaticamente o domínio dos dados
    - ✅ **Interpretação Inteligente:** IA entende perguntas em linguagem natural
    - ✅ **Execução Robusta:** Funções testadas e confiáveis para análises
    - ✅ **Insights Contextuais:** IA transforma resultados técnicos em valor de negócio
    - ✅ **Visualizações Garantidas:** Gráficos sempre funcionam (matplotlib/seaborn)
    - ✅ **Painel Completo:** Interface rica com análises rápidas
    - ✅ **Relatórios Inteligentes:** PDFs com insights da IA
    
    #### **🆚 Vantagem Competitiva:**
    
    | **Aspecto** | **Versão 1** | **Versão 2** | **🏆 Híbrida** |
    |-------------|--------------|--------------|-----------------|
    | **Interpretação** | Regras básicas | IA avançada | IA + fallback |
    | **Execução** | Sempre funciona | Instável | Sempre funciona |
    | **Visualizações** | Perfeitas | Falhavam | Perfeitas |
    | **Interface** | Completa | Simples | Completa |
    | **Insights** | Básicos | Inteligentes | Inteligentes |
    | **Confiabilidade** | Alta | Média | Máxima |
    
    #### **🔑 Como Usar:**
    
    1. **Configure Gemini (Gratuito):**
       - API Key gratuita do Google
       - Análise inteligente ativada
       - Fallback gracioso se não configurar
    
    2. **Carregue seu CSV:**
       - IA analisa automaticamente
       - Interface completa disponível
       - Sugestões inteligentes geradas
    
    3. **Faça Perguntas Naturais:**
       - "Quais os principais insights sobre fraude?"
       - "Mostre correlações mais importantes"
       - "Detecte outliers na coluna Amount"
    
    4. **Use Painel Rápido:**
       - Botões para análises instantâneas
       - Gráficos sempre funcionam
       - Cache para performance
    
    ### 🎓 **Para o Desafio I2A2:**
    
    **Framework:** Streamlit + Gemini + Funções Robustas
    **Diferencial:** Sistema híbrido que nunca falha
    **Genérico:** Funciona com qualquer CSV
    **Inteligente:** IA real interpretando dados
    **Completo:** Atende todos os requisitos
    
    ---
    
    🧠 **Powered by Google Gemini + Análises Robustas** | 🎯 **Híbrido = Confiável** | 🚀 **I2A2 Academy 2025**
    """)

# === RODAPÉ ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
🤖 <strong>Agente Híbrido: IA + Funções Robustas</strong><br>
🧠 Powered by Google Gemini + Análises Confiáveis<br>
Desenvolvido para o <strong>Desafio I2A2 Academy</strong> | Setembro 2025<br>
</div>
""", unsafe_allow_html=True)


