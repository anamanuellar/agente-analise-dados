import streamlit as st
import pandas as pd
import numpy as np
import io
import base64

# Importar todas as funções do utils.py
from utils import (
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

# Configuração da página
st.set_page_config(
    page_title="Agente de Análise de Dados Inteligente",
    page_icon="🤖",
    layout="wide"
)

# Inicializar memória do agente no session_state
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = {
        'conclusions': [],
        'insights': [],
        'patterns_found': [],
        'analysis_history': [],
        'generated_plots': []
    }

# Cache para evitar re-execução de análises
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Forçar reset do cache legado
for key in list(st.session_state.analysis_cache.keys()):
    if not isinstance(st.session_state.analysis_cache[key], tuple):
        del st.session_state.analysis_cache[key]

# Inicializar configurações avançadas se não existirem
if 'max_sample_size' not in st.session_state:
    st.session_state.max_sample_size = 5000
if 'contamination_rate' not in st.session_state:
    st.session_state.contamination_rate = 0.10

# Título principal
st.title("🤖 Agente Autônomo de Análise de Dados")
st.markdown("*Inteligente e Adaptativa com Memória, Análise Contextual e Exportação PDF*")
st.markdown("---")

# Sidebar para upload de arquivo
st.sidebar.header("📁 Upload do Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Escolha um arquivo CSV",
    type=['csv'],
    help="Faça upload de um arquivo CSV para análise automática"
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

# Sidebar - Configurações Avançadas
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações")
st.session_state.max_sample_size = st.sidebar.slider("Tamanho máximo da amostra para clustering:", 1000, 20000, st.session_state.max_sample_size)
st.session_state.contamination_rate = st.sidebar.slider("Taxa de contaminação para outliers:", 0.01, 0.20, st.session_state.contamination_rate)

# --- Lógica Principal da Aplicação ---

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset carregado com sucesso!")
        
        st.subheader("📊 Preview do Dataset")
        st.dataframe(df.head())

        # Informações do Dataset (movido para cá)
        st.markdown("---")
        st.subheader("📋 Informações Gerais do Dataset")
        dataset_info_text = get_dataset_info(df)
        st.markdown(dataset_info_text)

        # Campo de pergunta do usuário
        st.markdown("---")
        st.subheader("💬 Faça uma Pergunta ao Agente")
        user_question = st.text_input("Digite sua pergunta aqui:", key="user_question")

        # Sugestões Adaptativas (agora como texto)
        st.markdown("**💡 Sugestões de Perguntas:**")
        suggestions = get_adaptive_suggestions(df)
        for suggestion in suggestions:
            st.markdown(suggestion)
        st.markdown("---")

        # Processar a pergunta do usuário
        if user_question:
            st.write(f"🔍 **Pergunta interpretada como:** ")
            analysis_type = interpret_question(user_question, df)
            st.write(f"**{analysis_type.replace('_', ' ').title()}**")

            if analysis_type == 'distribution':
                st.subheader("📈 Análise de Distribuição Avançada")
                col_to_analyze = st.selectbox("Selecione a coluna para análise de distribuição:", df.select_dtypes(include=[np.number]).columns)
                if col_to_analyze:
                    with st.spinner(f'Gerando distribuição para {col_to_analyze}...'):
                        fig, conclusion = plot_distribution(df, col_to_analyze) # Agora retorna a figura e a conclusão
                        st.pyplot(fig)
                        st.info(conclusion) # Exibe a conclusão
            
            elif analysis_type == 'correlation':
                st.subheader("🔗 Análise de Correlação Avançada")
                with st.spinner('Calculando matriz de correlação...'):
                    fig, conclusion = plot_correlation_heatmap(df) # Agora retorna a figura e a conclusão
                    if fig:
                        st.pyplot(fig)
                        st.info(conclusion) # Exibe a conclusão
                    else:
                        st.error("❌ Necessário pelo menos 2 colunas numéricas para análise de correlação.")
            
            elif analysis_type == 'temporal':
                st.subheader("⏰ Análise de Padrões Temporais")
                time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if time_cols:
                    selected_time_col = st.selectbox("Selecione a coluna temporal:", time_cols)
                    if selected_time_col:
                        with st.spinner(f'Analisando padrões temporais para {selected_time_col}...'):
                            fig, conclusion = analyze_temporal_patterns(df, selected_time_col)
                            if fig:
                                st.pyplot(fig)
                            st.info(conclusion)
                else:
                    st.warning("❌ Nenhuma coluna temporal encontrada no dataset.")
            
            elif analysis_type == 'clustering':
                st.subheader("🎯 Análise Avançada de Clustering")
                if st.button("🎯 Executar Clustering Inteligente"):
                    with st.spinner('Executando análise de clustering...'):
                        fig, conclusion = perform_clustering_analysis(df, sample_size=st.session_state.max_sample_size) # Agora retorna a conclusão
                        st.info(conclusion) # Exibe a conclusão
            
            elif analysis_type == 'frequency':
                st.subheader("📊 Análise de Valores Frequentes")
                if st.button("🔍 Analisar Valores Mais/Menos Frequentes"):
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
                                
                                st.write(f"Valores únicos: {data['unique_count']}, Nulos: {data['null_count']}")
                                st.markdown("---")
                        else:
                            st.info("Nenhuma coluna categórica ou discreta encontrada para análise de frequência.")
            
            elif analysis_type == 'memory':
                st.subheader("🧠 Memória Completa do Agente")
                memory_summary = get_memory_summary()
                st.markdown(memory_summary)
                
                if st.session_state.agent_memory['conclusions']:
                    st.write("**📚 Histórico Detalhado de Análises:**")
                    for i, entry in enumerate(st.session_state.agent_memory['conclusions'], 1):
                        with st.expander(f"Análise {i}: {entry['analysis_type'].replace('_', ' ').title()}"):
                            st.write(f"**🔍 Conclusão:** {entry['conclusion']}")
                            st.write(f"**⏰ Timestamp:** {entry['timestamp']}")
                            if entry['data_info']:
                                st.write("**📊 Dados da Análise:**")
                                st.json(entry['data_info'])
            
            elif analysis_type == 'descriptive':
                st.subheader("📈 Análise Descritiva Completa")
                if st.button("📊 Gerar Estatísticas Descritivas Avançadas"):
                    with st.spinner('Gerando análise descritiva...'):
                        desc_stats = perform_descriptive_analysis(df)
                        st.dataframe(desc_stats, use_container_width=True)
                        
                        st.write("**📋 Informações Detalhadas do Dataset:**")
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        info_str = buffer.getvalue()
                        st.text(info_str)
            
            elif analysis_type == 'outliers':
                st.subheader("🔍 Detecção de Outliers Avançada")
                col_to_analyze = st.selectbox("Selecione a coluna para detecção de outliers:", df.select_dtypes(include=[np.number]).columns)
                if col_to_analyze:
                    if st.button(f"Detectar Outliers em {col_to_analyze}"):
                        with st.spinner(f'Detectando outliers em {col_to_analyze}...'):
                            fig, conclusion = detect_outliers(df, col_to_analyze) # Agora retorna a conclusão
                            st.info(conclusion) # Exibe a conclusão
            
            elif analysis_type == 'balance':
                st.subheader("⚖️ Análise de Balanceamento de Classes")
                binary_cols = [col for col in df.columns if df[col].nunique() == 2]
                if binary_cols:
                    selected_col = st.selectbox("Selecione a coluna binária para análise de balanceamento:", binary_cols)
                    if selected_col:
                        if st.button(f"Analisar Balanceamento de {selected_col}"):
                            with st.spinner(f'Analisando balanceamento da coluna {selected_col}...'):
                                fig, conclusion = analyze_balance(df, selected_col)
                                if fig:
                                    st.pyplot(fig)
                                st.info(conclusion)
                else:
                    st.warning("❌ Nenhuma coluna binária encontrada para análise de balanceamento.")

            else: # general ou não reconhecido
                st.info("Não entendi sua pergunta. Tente reformular ou use as sugestões.")
        
        # Painel de análises rápidas
        st.markdown("---")
        st.subheader("⚡ Painel de Análises Rápidas")
        
        col_buttons, col_results = st.columns([1, 2])
        
        with col_buttons:
            st.write("**Selecione uma análise:**")
            
            if st.button("📊 Estatísticas Descritivas", help="Análise descritiva completa", use_container_width=True):
                st.session_state.quick_analysis = "descriptive"
            
            if st.button("🔗 Mapa de Correlação", help="Correlações entre variáveis", use_container_width=True):
                st.session_state.quick_analysis = "correlation"
            
            if st.button("⏰ Padrões Temporais", help="Análise temporal (se disponível)", use_container_width=True):
                st.session_state.quick_analysis = "temporal"
            
            if st.button("🎯 Clustering Automático", help="Agrupamento inteligente", use_container_width=True):
                st.session_state.quick_analysis = "clustering"
            
            if st.button("📊 Valores Frequentes", help="Análise de frequências", use_container_width=True):
                st.session_state.quick_analysis = "frequency"
            
            if st.button("🧠 Memória do Agente", help="Histórico de análises", use_container_width=True):
                st.session_state.quick_analysis = "memory"
            
            if st.button("🔍 Detecção de Outliers", help="Identifica valores atípicos", use_container_width=True):
                st.session_state.quick_analysis = "outliers"
            
            if st.button("⚖️ Balanceamento de Classes", help="Análise de colunas binárias", use_container_width=True):
                st.session_state.quick_analysis = "balance"
        
        with col_results:
            if 'quick_analysis' in st.session_state:
                analysis_type = st.session_state.quick_analysis
                
                if analysis_type == "descriptive":
                    with st.spinner('Calculando estatísticas...'):
                        desc_stats = perform_descriptive_analysis(df)
                        st.dataframe(desc_stats, use_container_width=True)
                
                elif analysis_type == "correlation":
                    with st.spinner('Calculando correlações...'):
                        fig, conclusion = plot_correlation_heatmap(df) # Captura a conclusão
                        if fig:
                            st.pyplot(fig)
                            st.info(conclusion) # Exibe a conclusão
                        else:
                            st.error("Necessário pelo menos 2 colunas numéricas")
                
                elif analysis_type == "temporal":
                    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if time_cols:
                        with st.spinner('Analisando padrões temporais...'):
                            fig, message = analyze_temporal_patterns(df, time_cols[0]) # Usa a primeira coluna temporal encontrada
                            if fig:
                                st.pyplot(fig)
                            st.info(message)
                    else:
                        st.warning("Coluna temporal não encontrada")
                
                elif analysis_type == "clustering":
                    with st.spinner('Executando clustering...'):
                        fig, conclusion = perform_clustering_analysis(df, sample_size=st.session_state.max_sample_size) # Captura a conclusão
                        st.info(conclusion) # Exibe a conclusão
                
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
                
                elif analysis_type == "outliers":
                    st.info("Para detecção de outliers, use o campo de pergunta acima e especifique a coluna, ex: 'Detecte outliers na coluna Amount'")
                
                elif analysis_type == "balance":
                    st.info("Para análise de balanceamento, use o campo de pergunta acima e especifique a coluna, ex: 'Analise o balanceamento da coluna Class'")

            else:
                st.info("👈 Selecione uma análise rápida ao lado para ver os resultados aqui.")
        
        # Central de Inteligência do Agente
        st.markdown("---")
        st.subheader("🤖 Central de Inteligência do Agente")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Executar Resumo Das Análises", help="Gera resumo completo de todas as análises", type="primary", use_container_width=True):
                with st.spinner('Gerando resumo completo das análises...'):
                    complete_summary = generate_complete_analysis_summary(df)
                    st.markdown(complete_summary)
        
        with col2:
            if st.button("📄 Gerar Relatório PDF Completo", help="Exporta relatório com dataset e análises", type="secondary", use_container_width=True):
                if st.session_state.agent_memory['conclusions']:
                    with st.spinner('Gerando relatório PDF completo...'):
                        try:
                            pdf_content = generate_pdf_report(df)
                            
                            b64_pdf = base64.b64encode(pdf_content).decode()
                            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="relatorio_completo_analise_dados.pdf">📥 Clique aqui para baixar o relatório PDF</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            st.success("✅ Relatório PDF gerado com sucesso!")
                        except Exception as e:
                            st.error(f"❌ Erro ao gerar PDF: {str(e)}")
                else:
                    st.warning("⚠️ Execute o resumo das análises primeiro.")
    
    except Exception as e:
        st.error(f"❌ Erro ao processar o dataset: {str(e)}")
        st.write("**Detalhes do erro:**")
        st.code(str(e))

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
    - ✅ **Análise de Balanceamento:** Para colunas binárias
    
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
    - *"Detecte outliers na coluna Amount"*
    - *"Mostre valores mais frequentes"*
    - *"Qual sua memória de análises?"*
    - *"Analise o balanceamento da coluna Class"*
    
    ---
    
    
    ✅ **Memória do Agente** | ✅ **Padrões Temporais** | ✅ **Clustering** | ✅ **Exportação PDF** | ✅ **Inteligência Adaptativa**
    """)


# Rodapé
st.markdown("---")
st.markdown("""
<div style=\'text-align: center; color: #666; font-size: 14px;\'>
🤖 <strong>Agente Autônomo de Análise de Dados</strong>  

Desenvolvido para o <strong>Desafio I2A2 Academy</strong> por Ana Manuella Ribeiro | Setembro 2025  

</div>
""", unsafe_allow_html=True)
