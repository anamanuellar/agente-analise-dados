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
        """Análise inicial PROFUNDA e analítica do dataset usando Gemini"""
        
        # Primeiro, calcular estatísticas avançadas para o prompt
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        
        # Análise de correlações se houver colunas numéricas
        correlation_summary = ""
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.5:
                        high_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
            correlation_summary = f"Correlações significativas encontradas: {len(high_corr)} pares com |r| > 0.5"
        
        # Análise de distribuições
        distribution_summary = ""
        if len(numeric_cols) > 0:
            skew_analysis = []
            for col in numeric_cols[:5]:  # Primeiras 5 colunas
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    skew_analysis.append(f"{col}: assimetria alta ({skewness:.2f})")
                elif abs(skewness) > 0.5:
                    skew_analysis.append(f"{col}: assimetria moderada ({skewness:.2f})")
            distribution_summary = f"Distribuições: {'; '.join(skew_analysis) if skew_analysis else 'distribuições aproximadamente normais'}"
        
        # Análise de qualidade detalhada
        missing_analysis = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                missing_analysis.append(f"{col}: {missing_pct:.1f}% faltante")
        
        system_context = """Você é um Senior Data Scientist com 15+ anos de experiência em análise exploratória de dados. 
        Sua especialidade é identificar padrões, anomalias e oportunidades de insight em datasets complexos.
        Forneça análises TÉCNICAS, ESPECÍFICAS e QUANTITATIVAS. Evite respostas genéricas."""
        
        prompt = f"""
        DATASET PARA ANÁLISE PROFUNDA:

        📊 ESTRUTURA:
        - {df.shape[0]:,} registros × {df.shape[1]} variáveis
        - Densidade: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%
        - Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

        📈 COMPOSIÇÃO DOS DADOS:
        - Numéricas: {list(numeric_cols)} ({len(numeric_cols)} colunas)
        - Categóricas: {list(categorical_cols)} ({len(categorical_cols)} colunas)

        📋 AMOSTRA REPRESENTATIVA:
        {df.head(3).to_string()}

        📊 ESTATÍSTICAS QUANTITATIVAS:
        {df.describe().to_string() if len(numeric_cols) > 0 else "Sem colunas numéricas para análise estatística"}

        🔗 ANÁLISE DE CORRELAÇÕES:
        {correlation_summary}

        📐 ANÁLISE DE DISTRIBUIÇÕES:
        {distribution_summary}

        🔍 QUALIDADE DOS DADOS:
        - Valores faltantes: {'; '.join(missing_analysis[:10]) if missing_analysis else 'Dataset completo'}
        - Duplicatas: {df.duplicated().sum():,} registros
        # Calcular primeiro, depois formatar:
        variability_text = "N/A"
        if len(numeric_cols) > 0:
            try:
                cv_mean = (df[numeric_cols].std() / df[numeric_cols].mean()).mean()
                if not np.isnan(cv_mean) and np.isfinite(cv_mean):
                    variability_text = f"{cv_mean:.3f}"
            except:
                variability_text = "N/A"
        - Variabilidade: CV médio = {variability_text}

        FORNEÇA UMA ANÁLISE ESTRUTURADA E TÉCNICA:

        ## IDENTIFICAÇÃO DO DOMÍNIO
        Com base nos nomes das colunas, distribuições e padrões, identifique especificamente o tipo de dataset (ex: transações financeiras, dados de marketing, logs de sistema, etc.) e justifique sua conclusão.

        ## AVALIAÇÃO TÉCNICA DE QUALIDADE
        Avalie objetivamente: completude, consistência, outliers potenciais, balanceamento (se aplicável). Use números específicos.

        ## CARACTERÍSTICAS ANALÍTICAS PRINCIPAIS  
        - [Característica 1: padrão específico identificado com evidências]
        - [Característica 2: distribuição ou correlação relevante]
        - [Característica 3: aspecto de qualidade ou estrutura importante]

        ## ANÁLISES PRIORITÁRIAS RECOMENDADAS
        - [Análise 1: técnica específica e por que é crítica para este dataset]
        - [Análise 2: método estatístico recomendado e valor esperado]
        - [Análise 3: exploração direcionada baseada nos padrões identificados]

        ## HIPÓTESES E INSIGHTS POTENCIAIS
        - [Hipótese 1: baseada em evidências dos dados observados]
        - [Hipótese 2: padrão ou anomalia que merece investigação]
        - [Hipótese 3: oportunidade de descoberta específica]

        Seja TÉCNICO, ESPECÍFICO e baseado em EVIDÊNCIAS dos dados mostrados.
        """
        
        response = self._call_gemini(prompt, system_context)
        
        # Parsing mais robusto da resposta
        analysis_result = self._parse_detailed_analysis(response, df)
        
        self.dataset_context = analysis_result
        self._add_to_gemini_memory("initial_analysis", analysis_result)
        
        return analysis_result
    
    def _parse_detailed_analysis(self, response: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse robusto da análise detalhada"""
        try:
            # Extrair seções com melhor parsing
            dataset_type = self._extract_robust_section(response, ["IDENTIFICAÇÃO DO DOMÍNIO", "DOMÍNIO"], ["AVALIAÇÃO", "QUALIDADE"])
            data_quality = self._extract_robust_section(response, ["AVALIAÇÃO", "QUALIDADE"], ["CARACTERÍSTICAS", "ANALÍTICAS"])
            key_characteristics = self._extract_robust_list(response, ["CARACTERÍSTICAS", "ANALÍTICAS"], ["ANÁLISES", "PRIORITÁRIAS", "RECOMENDADAS"])
            recommended_analyses = self._extract_robust_list(response, ["ANÁLISES", "RECOMENDADAS", "PRIORITÁRIAS"], ["HIPÓTESES", "INSIGHTS"])
            potential_insights = self._extract_robust_list(response, ["HIPÓTESES", "INSIGHTS"], None)
            
            # Garantir conteúdo mínimo de qualidade
            if not dataset_type or len(dataset_type.strip()) < 10:
                dataset_type = self._generate_fallback_domain_analysis(df)
            
            if not key_characteristics or len(key_characteristics) < 2:
                key_characteristics = self._generate_fallback_characteristics(df)
            
            if not recommended_analyses or len(recommended_analyses) < 2:
                recommended_analyses = self._generate_fallback_recommendations(df)
                
            if not potential_insights or len(potential_insights) < 2:
                potential_insights = self._generate_fallback_insights(df)
            
            return {
                "dataset_type": dataset_type.strip(),
                "data_quality": data_quality.strip() or self._generate_fallback_quality_analysis(df),
                "key_characteristics": key_characteristics,
                "recommended_analyses": recommended_analyses,
                "potential_insights": potential_insights,
                "full_response": response
            }
            
        except Exception as e:
            # Fallback com análise robusta própria
            return {
                "dataset_type": self._generate_fallback_domain_analysis(df),
                "data_quality": self._generate_fallback_quality_analysis(df),
                "key_characteristics": self._generate_fallback_characteristics(df),
                "recommended_analyses": self._generate_fallback_recommendations(df),
                "potential_insights": self._generate_fallback_insights(df),
                "full_response": response
            }
    
    def _extract_robust_section(self, text: str, start_markers: List[str], end_markers: List[str]) -> str:
        """Extração robusta de seções com múltiplos marcadores"""
        for start_marker in start_markers:
            try:
                start_idx = text.upper().find(start_marker.upper())
                if start_idx != -1:
                    start_idx = start_idx + len(start_marker)
                    
                    end_idx = len(text)
                    if end_markers:
                        for end_marker in end_markers:
                            temp_end = text.upper().find(end_marker.upper(), start_idx)
                            if temp_end != -1 and temp_end < end_idx:
                                end_idx = temp_end
                    
                    section = text[start_idx:end_idx].strip()
                    # Limpar marcadores markdown
                    section = section.replace("##", "").replace("#", "").strip()
                    
                    if len(section) > 10:  # Seção substancial
                        return section
            except:
                continue
        return ""
    
    def _extract_robust_list(self, text: str, start_markers: List[str], end_markers: List[str]) -> List[str]:
        """Extração robusta de listas"""
        section = self._extract_robust_section(text, start_markers, end_markers)
        if not section:
            return []
        
        items = []
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
                item = line.lstrip('-•*123456789. ').strip()
                if len(item) > 15:  # Item substancial
                    items.append(item)
        
        return items[:5]  # Máximo 5 itens
    
    def _generate_fallback_domain_analysis(self, df: pd.DataFrame) -> str:
        """Gera análise de domínio robusta baseada nas colunas"""
        cols_lower = [col.lower() for col in df.columns]
        
        # Detecção por padrões de nomes
        if any(word in ' '.join(cols_lower) for word in ['transaction', 'amount', 'fraud', 'class']):
            return "Dataset de Detecção de Fraude - baseado em colunas de transação e classificação"
        elif any(word in ' '.join(cols_lower) for word in ['price', 'sales', 'revenue', 'customer']):
            return "Dataset de Vendas/E-commerce - baseado em variáveis comerciais"
        elif any(word in ' '.join(cols_lower) for word in ['time', 'timestamp', 'date']):
            return "Dataset Temporal - contém componentes de série temporal"
        elif len(df.select_dtypes(include=[np.number]).columns) > len(df.columns) * 0.7:
            return "Dataset Quantitativo - predominantemente numérico para análise estatística"
        else:
            return f"Dataset Misto - {len(df.select_dtypes(include=[np.number]).columns)} variáveis numéricas e {len(df.select_dtypes(include=['object']).columns)} categóricas"
    
    def _generate_fallback_quality_analysis(self, df: pd.DataFrame) -> str:
        """Gera análise de qualidade robusta"""
        completeness = ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100)
        duplicates = df.duplicated().sum()
        
        quality_score = "EXCELENTE" if completeness > 95 and duplicates == 0 else \
                       "BOA" if completeness > 90 else \
                       "MODERADA" if completeness > 80 else "BAIXA"
        
        return f"Qualidade {quality_score}: {completeness:.1f}% completo, {duplicates:,} duplicatas, {df.shape[0]:,} registros válidos para análise"
    
    def _generate_fallback_characteristics(self, df: pd.DataFrame) -> List[str]:
        """Gera características robustas baseadas na análise dos dados"""
        characteristics = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            skew_analysis = df[numeric_cols].skew()
            highly_skewed = sum(abs(skew_analysis) > 1)
            characteristics.append(f"Distribuições: {highly_skewed} de {len(numeric_cols)} variáveis numéricas apresentam assimetria alta")
            
            # Análise de variabilidade
            cv = (df[numeric_cols].std() / df[numeric_cols].mean()).mean()
            if cv > 1:
                characteristics.append(f"Variabilidade alta: coeficiente de variação médio de {cv:.2f} indica dados heterogêneos")
            
            # Análise de correlações
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                high_corr_count = sum(sum(abs(corr_matrix.values) > 0.7) - len(corr_matrix)) // 2
                characteristics.append(f"Estrutura de correlação: {high_corr_count} pares de variáveis altamente correlacionadas")
        
        # Análise categórica
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            high_cardinality = sum(df[col].nunique() > 50 for col in categorical_cols)
            if high_cardinality > 0:
                characteristics.append(f"Cardinalidade: {high_cardinality} variáveis categóricas com alta diversidade (>50 valores únicos)")
        
        return characteristics[:3] if characteristics else ["Dataset com estrutura padrão para análise exploratória"]
    
    def _generate_fallback_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Gera recomendações robustas baseadas na estrutura dos dados"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            recommendations.append(f"Análise de correlação entre {len(numeric_cols)} variáveis numéricas para identificar relacionamentos lineares")
        
        if len(numeric_cols) > 0:
            recommendations.append("Detecção de outliers multivariada usando Isolation Forest para identificar anomalias")
        
        if df.shape[0] > 1000:
            recommendations.append("Clustering hierárquico ou K-means para segmentação e identificação de padrões latentes")
        
        # Análise temporal se detectada
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            recommendations.append(f"Análise de séries temporais na coluna {time_cols[0]} para identificar tendências e sazonalidade")
        
        return recommendations[:3] if recommendations else ["Análise exploratória sistemática das variáveis principais"]
    
    def _generate_fallback_insights(self, df: pd.DataFrame) -> List[str]:
        """Gera insights potenciais robustos"""
        insights = []
        
        # Insights baseados na estrutura
        if df.shape[0] > 10000:
            insights.append(f"Grande volume de dados ({df.shape[0]:,} registros) permite análises estatísticas robustas e modelagem preditiva")
        
        # Insights sobre balanceamento
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:
            value_counts = df[col].value_counts()
            if len(value_counts) == 2:  # Binária
                balance_ratio = value_counts.min() / value_counts.max()
                if balance_ratio < 0.1:
                    insights.append(f"Forte desbalanceamento na variável {col} ({balance_ratio:.2f}) sugere necessidade de técnicas de balanceamento")
        
        # Insights sobre missing data
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if len(missing_cols) > df.shape[1] * 0.3:
            insights.append(f"Padrão de dados faltantes em {len(missing_cols)} variáveis pode indicar processo de coleta estruturado")
        
        return insights[:3] if insights else ["Potencial para descoberta de padrões não óbvios através de análise multivariada"]
    
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

