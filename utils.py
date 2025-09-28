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

class GeminiAgent:
    """Agente que USA Google Gemini como cérebro do sistema"""
    
    def __init__(self, model_name="gemini-2.0-flash-exp"):  # Modelo mais recente
        self.model_name = model_name
        self.model = None
        self.conversation_history = []
        self.dataset_context = {}
        
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
        
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4000,
        }
    
    def configure_gemini(self, api_key: str):
        """Configura o Gemini com a API Key fornecida."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            self.model_name,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config
        )

    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        """Chama o Google Gemini para análise"""
        if not self.model:
            return self._fallback_response("Gemini não configurado. Por favor, forneça a API Key.")

        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            st.error(f"Erro no Gemini: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Resposta de fallback quando Gemini falha"""
        return f"""
**[Modo Fallback - Gemini Temporariamente Indisponível]**

Recebi sua pergunta: "{prompt}"

Esta é uma resposta de fallback. Para análise completa com IA:
1. Verifique sua API key do Gemini
2. Confirme conexão com internet
3. Tente novamente em alguns instantes

As funcionalidades básicas de análise continuam funcionando normalmente.
        """
    
    def analyze_dataset_initially(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise inicial inteligente do dataset usando Gemini"""
        
        system_context = """Você é um especialista sênior em análise de dados com PhD em Estatística e vasta experiência em Business Intelligence. 
        Analise datasets CSV e forneça insights profissionais, práticos e actionables."""
        
        prompt = f"""
        Analise este dataset CSV e forneça uma análise estruturada seguindo EXATAMENTE este formato:

        INFORMAÇÕES ESTRUTURAIS:
        - Linhas: {df.shape[0]:,}
        - Colunas: {df.shape[1]}
        - Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

        TIPOS DE DADOS:
        - Numéricas: {len(df.select_dtypes(include=[np.number]).columns)} colunas
        - Categóricas: {len(df.select_dtypes(include=["object"]).columns)} colunas
        - Dados faltantes: {df.isnull().sum().sum():,} valores

        NOMES DAS COLUNAS:
        {list(df.columns)}

        AMOSTRA DOS DADOS (3 primeiras linhas):
        {df.head(3).to_string()}

        ESTATÍSTICAS BÁSICAS (colunas numéricas):
        {df.select_dtypes(include=[np.number]).describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "Nenhuma coluna numérica"}

        Com base nessas informações, forneça uma análise estruturada seguindo EXATAMENTE este formato:

        ## IDENTIFICAÇÃO DO DOMÍNIO
        [Identifique que tipo de dataset é este: financeiro, marketing, fraude, vendas, saúde, etc.]

        ## AVALIAÇÃO DE QUALIDADE
        [Como você avalia a qualidade dos dados? Completude, consistência, etc.]

        ## CARACTERÍSTICAS PRINCIPAIS
        - [Característica 1: descrição]
        - [Característica 2: descrição]
        - [Característica 3: descrição]

        ## ANÁLISES RECOMENDADAS
        - [Análise 1: por que é importante]
        - [Análise 2: por que é importante]
        - [Análise 3: por que é importante]

        ## INSIGHTS POTENCIAIS
        - [Insight 1: o que pode ser descoberto]
        - [Insight 2: o que pode ser descoberto]
        - [Insight 3: o que pode ser descoberto]

        Seja específico, técnico mas acessível, focando em valor de negócio.
        """
        
        response = self._call_gemini(prompt, system_context)
        
        # Processar resposta estruturada
        try:
            dataset_type = self._extract_section(response, "IDENTIFICAÇÃO DO DOMÍNIO", "AVALIAÇÃO DE QUALIDADE")
            data_quality = self._extract_section(response, "AVALIAÇÃO DE QUALIDADE", "CARACTERÍSTICAS PRINCIPAIS")
            key_characteristics = self._extract_list_items(response, "CARACTERÍSTICAS PRINCIPAIS", "ANÁLISES RECOMENDADAS")
            recommended_analyses = self._extract_list_items(response, "ANÁLISES RECOMENDADAS", "INSIGHTS POTENCIAIS")
            potential_insights = self._extract_list_items(response, "INSIGHTS POTENCIAIS", None)
            
            analysis_result = {
                "dataset_type": dataset_type.strip() or "Dataset genérico identificado",
                "data_quality": data_quality.strip() or "Qualidade dos dados avaliada",
                "key_characteristics": key_characteristics or ["Análise detalhada disponível"],
                "recommended_analyses": recommended_analyses or ["Estatísticas descritivas", "Análise de correlação"],
                "potential_insights": potential_insights or ["Converse com o agente para descobrir insights"],
                "full_response": response
            }
            
        except Exception as e:
            analysis_result = {
                "dataset_type": "Dataset não classificado pelo Gemini",
                "data_quality": "Avaliação detalhada disponível via chat",
                "key_characteristics": ["Use o chat para análise detalhada"],
                "recommended_analyses": ["Análise descritiva", "Correlações", "Detecção de outliers"],
                "potential_insights": ["Converse com o Gemini para descobrir"],
                "full_response": response
            }
        
        self.dataset_context = analysis_result
        self._add_to_gemini_memory("initial_analysis", analysis_result)
        
        return analysis_result
    
    def process_user_query(self, user_query: str, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Processa query do usuário usando Gemini para interpretação e execução"""
        
        self.conversation_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now()
        })
        
        # FASE 1: GEMINI INTERPRETA A QUERY E CRIA PLANO
        analysis_plan = self._gemini_create_analysis_plan(user_query, df)
        
        # FASE 2: EXECUTAR ANÁLISE BASEADA NO PLANO
        analysis_results = self._execute_gemini_analysis_plan(analysis_plan, df)
        
        # FASE 3: GEMINI GERA RESPOSTA FINAL
        final_response = self._gemini_generate_final_response(user_query, analysis_plan, analysis_results, df)
        
        # FASE 4: CRIAR VISUALIZAÇÃO SE NECESSÁRIO - CORRIGIDO
        visualization = self._create_visualization_if_needed(analysis_plan, df, analysis_results)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response,
            "analysis_plan": analysis_plan,
            "results": analysis_results,
            "visualization": visualization,
            "timestamp": datetime.now()
        })
        
        self._add_to_gemini_memory("user_query", {
            "query": user_query,
            "response": final_response,
            "plan": analysis_plan,
            "results": analysis_results,
            "visualization": visualization
        })
        
        return final_response, visualization
    
    def _gemini_create_analysis_plan(self, user_query: str, df: pd.DataFrame) -> Dict:
        """Gemini cria plano de análise estruturado"""
        
        system_context = """Você é um agente especialista em análise de dados. Interprete perguntas do usuário e crie planos estruturados de análise em formato JSON."""
        
        prompt = f"""
        **Pergunta do Usuário:** "{user_query}"

        **Contexto do Dataset:**
        - Colunas disponíveis: {list(df.columns)}
        - Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)}
        - Colunas categóricas: {list(df.select_dtypes(include=["object"]).columns)}

        **Sua Tarefa:**
        Crie um plano de análise em formato JSON para responder à pergunta do usuário. O JSON deve ter a seguinte estrutura:
        {{
            "analysis_type": "[tipo_da_analise]",
            "columns_to_use": ["coluna1", "coluna2"],
            "requires_visualization": [true/false],
            "visualization_type": "[tipo_do_grafico]"
        }}

        **Tipos de Análise Válidos:**
        - `descriptive_statistics`: Para perguntas sobre média, mediana, desvio padrão, etc.
        - `correlation_matrix`: Para perguntas sobre correlação entre variáveis.
        - `distribution_plot`: Para perguntas sobre a distribuição de uma variável (histograma).
        - `outlier_detection`: Para perguntas sobre outliers ou valores atípicos.
        - `clustering`: Para perguntas sobre agrupamentos ou segmentação.
        - `frequency_analysis`: Para perguntas sobre valores mais/menos frequentes.
        - `temporal_analysis`: Para perguntas sobre padrões temporais.
        - `balance_analysis`: Para perguntas sobre balanceamento de classes.
        - `general_query`: Para perguntas gerais que não se encaixam nas categorias acima.

        **Tipos de Gráfico Válidos:**
        - `histogram`: Para `distribution_plot`.
        - `heatmap`: Para `correlation_matrix`.
        - `boxplot`: Para `outlier_detection`.
        - `scatterplot`: Para `clustering`.
        - `pie_chart`: Para `balance_analysis`.
        - `line_chart`: Para `temporal_analysis`.
        - `bar_chart`: Para `frequency_analysis`.

        **Exemplo de Resposta:**
        {{
            "analysis_type": "correlation_matrix",
            "columns_to_use": {list(df.select_dtypes(include=[np.number]).columns)[:2]},
            "requires_visualization": true,
            "visualization_type": "heatmap"
        }}

        **Sua Resposta (apenas o JSON):**
        """
        
        response = self._call_gemini(prompt, system_context)
        
        try:
            json_response = response.strip().replace("```json", "").replace("```", "").strip()
            plan = json.loads(json_response)
        except (json.JSONDecodeError, KeyError):
            plan = {
                "analysis_type": "general_query",
                "columns_to_use": [],
                "requires_visualization": False,
                "visualization_type": "none"
            }
        
        return plan

    def _execute_gemini_analysis_plan(self, plan: Dict, df: pd.DataFrame) -> Dict:
        """Executa o plano de análise gerando e executando código com Gemini."""
        
        analysis_type = plan.get("analysis_type", "general_query")
        columns = plan.get("columns_to_use", [])
        
        system_context = """Você é um especialista em Python para análise de dados. Gere código Python para realizar a análise solicitada. O código deve ser completo, funcional e imprimir os resultados em formato de texto."""
        
        prompt = f"""
        **Plano de Análise:**
        - Tipo: {analysis_type}
        - Colunas: {columns}

        **Sua Tarefa:**
        Gere o código Python completo para realizar esta análise no DataFrame `df`. O código deve:
        1. Usar as bibliotecas `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.
        2. Realizar a análise solicitada nas colunas especificadas.
        3. Imprimir os resultados da análise em formato de texto claro e informativo.
        4. Não gere o código para criar gráficos, apenas a análise textual.
        5. Se a análise for de clustering, use uma amostra de no máximo 5000 linhas para performance.
        6. Se a análise for de outliers, use IsolationForest, IQR e Z-score.

        **Exemplo de Código para `descriptive_statistics`:**
        ```python
        desc_stats = df["{columns[0] if columns else 'Amount'}"].describe()
        print("Estatísticas Descritivas:\\n", desc_stats)
        ```

        **Seu Código Python (apenas o código):**
        """
        
        code_to_execute = self._call_gemini(prompt, system_context)
        
        code_to_execute = code_to_execute.strip().replace("```python", "").replace("```", "").strip()
        
        try:
            output_buffer = io.StringIO()
            exec_globals = {
                "df": df,
                "pd": pd,
                "np": np,
                "plt": plt,
                "sns": sns,
                "StandardScaler": StandardScaler,
                "IsolationForest": IsolationForest,
                "KMeans": KMeans,
                "silhouette_score": silhouette_score,
                "print": lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)
            }
            
            exec(code_to_execute, exec_globals)
            
            text_output = output_buffer.getvalue()
            
            return {"status": "success", "text_output": text_output, "code_executed": code_to_execute}
            
        except Exception as e:
            return {"status": "error", "error_message": str(e), "code_executed": code_to_execute}

    def _gemini_generate_final_response(self, user_query: str, plan: Dict, results: Dict, df: pd.DataFrame) -> str:
        """Gera a resposta final em linguagem natural com base nos resultados."""
        
        system_context = """Você é um especialista em análise de dados apresentando resultados para um cliente. Seja claro, conciso e foque em insights de negócio."""
        
        prompt = f"""
        **Pergunta do Cliente:** "{user_query}"

        **Análise Realizada:**
        - Tipo: {plan.get("analysis_type")}
        - Colunas: {plan.get("columns_to_use")}

        **Resultados da Análise:**
        ```
        {results.get("text_output")}
        ```

        **Sua Tarefa:**
        Com base nos resultados, escreva uma resposta clara e informativa para o cliente. A resposta deve:
        1. Explicar o que foi analisado.
        2. Resumir os principais resultados.
        3. Fornecer insights práticos e recomendações.
        4. Usar linguagem acessível, evitando jargões técnicos excessivos.
        
        **Sua Resposta:**
        """
        
        return self._call_gemini(prompt, system_context)

    def _create_visualization_if_needed(self, plan: Dict, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """VERSÃO CORRIGIDA - Gera visualização funcionalmente"""
        
        if not plan.get("requires_visualization"): 
            return None
        
        visualization_type = plan.get("visualization_type")
        columns = plan.get("columns_to_use")
        
        system_context = """Você é um especialista em visualização de dados com Python. 
        Gere código Python completo e funcional para criar o gráfico solicitado.
        O código deve:
        1. Ser sintaticamente correto
        2. Criar uma figura usando plt.subplots()
        3. Usar matplotlib/seaborn para o gráfico
        4. Incluir título e labels
        5. Armazenar a figura na variável 'generated_fig'
        6. Usar plt.tight_layout()
        """
        
        prompt = f"""
        **Plano de Visualização:**
        - Tipo de Gráfico: {visualization_type}
        - Colunas: {columns}

        **Sua Tarefa:**
        Gere código Python completo para criar este gráfico. O código deve:

        1. Criar uma figura: `generated_fig, ax = plt.subplots(figsize=(10, 6))`
        2. Gerar o gráfico solicitado no eixo `ax`
        3. Incluir título e rótulos
        4. Terminar com `plt.tight_layout()`
        5. A figura deve estar na variável `generated_fig`

        **Exemplo para histogram:**
        ```python
        generated_fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['{columns[0] if columns else 'Amount'}'], kde=True, ax=ax)
        ax.set_title('Distribuição de {columns[0] if columns else 'Amount'}')
        ax.set_xlabel('{columns[0] if columns else 'Amount'}')
        ax.set_ylabel('Frequência')
        plt.tight_layout()
        ```

        **Seu Código Python (apenas o código, sem comentários):**
        """
        
        code_to_execute = self._call_gemini(prompt, system_context)
        
        # Limpar o código
        code_to_execute = code_to_execute.strip().replace("```python", "").replace("```", "").strip()
        
        try:
            # CORREÇÃO PRINCIPAL: usar exec() e depois capturar a variável
            exec_globals = {
                "df": df,
                "pd": pd,
                "np": np,
                "plt": plt,
                "sns": sns,
                "StandardScaler": StandardScaler,
                "IsolationForest": IsolationForest,
                "KMeans": KMeans,
                "silhouette_score": silhouette_score
            }
            
            # Executar o código
            exec(code_to_execute, exec_globals)
            
            # Capturar a figura gerada
            generated_fig = exec_globals.get('generated_fig')
            
            if generated_fig is None:
                return {"status": "error", "error_message": "Figura não foi criada corretamente", "code_executed": code_to_execute}
            
            return {"status": "success", "figure": generated_fig, "code_executed": code_to_execute}
            
        except Exception as e:
            st.error(f"Erro ao gerar visualização: {e}")
            st.code(code_to_execute, language="python")
            return {"status": "error", "error_message": str(e), "code_executed": code_to_execute}

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extrai seção de texto entre dois marcadores."""
        try:
            start_index = text.index(start_marker) + len(start_marker)
            if end_marker:
                end_index = text.index(end_marker, start_index)
                return text[start_index:end_index].strip()
            else:
                return text[start_index:].strip()
        except ValueError:
            return ""

    def _extract_list_items(self, text: str, start_marker: str, end_marker: str) -> List[str]:
        """Extrai itens de lista de uma seção de texto."""
        section = self._extract_section(text, start_marker, end_marker)
        return [item.strip().lstrip("- ") for item in section.split("\n") if item.strip().startswith("-")]

    def _add_to_gemini_memory(self, analysis_type: str, data: Dict):
        """Adiciona análise à memória do agente Gemini."""
        if "gemini_memory" not in st.session_state:
            st.session_state.gemini_memory = []
        
        st.session_state.gemini_memory.append({
            "type": analysis_type,
            "data": data,
            "timestamp": datetime.now()
        })

    def get_full_memory_summary(self) -> str:
        """Gera um resumo completo de todas as interações e análises na memória do Gemini."""
        summary_text = """
🤖 **RESUMO COMPLETO DAS ANÁLISES - AGENTE AUTÔNOMO COM GEMINI**

"""
        if not st.session_state.gemini_memory:
            return summary_text + "Nenhuma análise realizada ainda."

        for i, entry in enumerate(st.session_state.gemini_memory):
            summary_text += f"\n--- **Análise {i+1}: {entry['type'].replace('_', ' ').title()}** ---\n"
            summary_text += f"**Timestamp:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if entry["type"] == "initial_analysis":
                summary_text += f"**Análise Inicial do Dataset:**\n{entry['data']['full_response']}\n"
            elif entry["type"] == "user_query":
                summary_text += f"**Pergunta do Usuário:** {entry['data']['query']}\n"
                summary_text += f"**Plano de Análise:** {json.dumps(entry['data']['plan'], indent=2)}\n"
                summary_text += f"**Resultados da Execução:**\n```\n{entry['data']['results']['text_output']}\n```\n"
                summary_text += f"**Resposta Final do Gemini:**\n{entry['data']['response']}\n"
            
        return summary_text

    def generate_smart_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Gera sugestões inteligentes baseadas no dataset usando LLM"""
        
        system_context = """Você é um especialista em análise de dados. Gere 5-6 sugestões específicas e práticas de perguntas que o usuário pode fazer sobre seus dados."""
        
        user_prompt = f"""
        DATASET CONTEXT:
        - Tipo: {self.dataset_context.get("dataset_type", "Genérico")}
        - Shape: {df.shape[0]:,} linhas x {df.shape[1]} colunas
        - Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)[:8]}
        - Colunas categóricas: {list(df.select_dtypes(include=["object"]).columns)[:8]}
        
        ANÁLISES JÁ REALIZADAS:
        {len(self.conversation_history)} interações anteriores
        
        Gere 5-6 sugestões específicas de perguntas que seriam valiosas para este dataset.
        Formato: "Pergunta específica e clara"
        
        Exemplo de formato:
        - "Quais são as correlações mais fortes entre as variáveis numéricas?"
        - "Existem outliers significativos na coluna Amount?"
        """
        
        response = self._call_gemini(user_prompt, system_context)
        
        suggestions = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                suggestion = line[1:].strip().strip("\"")
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:6]

# Funções auxiliares mantidas do código original
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
- Categóricas: {len(df.select_dtypes(include=["object", "category"]).columns)}
- Datetime: {len(df.select_dtypes(include=["datetime"]).columns)}

**Estatísticas Gerais:**
- Densidade de dados: {(df.count().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%
- Variabilidade média: {df.select_dtypes(include=[np.number]).std().mean():.2f}
"""
    return info

def generate_pdf_report(df, agent: GeminiAgent):
    """Gera relatório PDF com todas as análises e informações do dataset."""
    pdf_buffer = io.BytesIO()
    
    with PdfPages(pdf_buffer) as pdf:
        # Página 1: Capa e informações do dataset
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.text(0.5, 0.95, "🤖 RELATÓRIO COMPLETO DE ANÁLISE DE DADOS", 
                ha="center", va="top", fontsize=16, fontweight="bold")
        ax.text(0.5, 0.90, "Agente Autônomo com IA Generativa", 
                ha="center", va="top", fontsize=12)
        ax.text(0.5, 0.87, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                ha="center", va="top", fontsize=10)
        
        dataset_info = get_dataset_info(df)
        ax.text(0.05, 0.80, dataset_info, ha="left", va="top", fontsize=8, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0f7fa", edgecolor="#00bcd4"))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Páginas subsequentes com análises
        if "gemini_memory" in st.session_state and st.session_state.gemini_memory:
            for i, entry in enumerate(st.session_state.gemini_memory):
                fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                ax = fig.add_subplot(111)
                
                title = f"Análise {i+1}: {entry['type'].replace('_', ' ').title()}"
                ax.text(0.05, 0.95, title, ha="left", va="top", fontsize=14, fontweight="bold")
                
                content = ""
                if entry["type"] == "initial_analysis":
                    content = entry["data"]["full_response"]
                elif entry["type"] == "user_query":
                    content = f"Pergunta: {entry['data']['query']}\n\n" \
                              f"Resposta Final:\n{entry['data']['response']}"
                
                ax.text(0.05, 0.90, content[:1500], ha="left", va="top", fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Adicionar gráficos se existirem
                if (entry["type"] == "user_query" and 
                    entry["data"]["visualization"] and 
                    entry["data"]["visualization"]["status"] == "success" and
                    entry["data"]["visualization"]["figure"]):
                    
                    fig_vis = entry["data"]["visualization"]["figure"]
                    pdf.savefig(fig_vis, bbox_inches="tight")
                    plt.close(fig_vis)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# Função para inicializar o agente Gemini
def initialize_gemini_agent():
    """Inicializa o agente Gemini no Streamlit"""
    if 'gemini_agent' not in st.session_state:
        st.session_state.gemini_agent = GeminiAgent()
    
    return st.session_state.gemini_agent

# Função de compatibilidade para o app.py
def get_adaptive_suggestions(df):
    """Função de compatibilidade - usa o agente Gemini se disponível"""
    if 'gemini_agent' in st.session_state and st.session_state.gemini_agent.model:
        return st.session_state.gemini_agent.generate_smart_suggestions(df)
    else:
        # Fallback para sugestões básicas
        return [
            "Mostre estatísticas descritivas das colunas numéricas",
            "Analise correlações entre as variáveis",
            "Detecte outliers nos dados",
            "Faça clustering automático",
            "Mostre a distribuição da coluna principal",
            "Qual a memória do agente?"
        ]
