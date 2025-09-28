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

# Importação do PdfPages com verificação
try:
    from matplotlib.backends.backend_pdf import PdfPages
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class GeminiAgent:
    """Agente que USA Google Gemini como cérebro do sistema"""
    
    def __init__(self, model_name="gemini-2.0-flash-exp"):
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
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        
        # Análise de correlações
        correlation_summary = ""
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_val) and abs(corr_val) > 0.5:
                            high_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
                correlation_summary = f"Correlações significativas encontradas: {len(high_corr)} pares com |r| > 0.5"
            except:
                correlation_summary = "Correlações: análise requer limpeza de dados"
        else:
            correlation_summary = "Correlações: insuficientes colunas numéricas"
        
        # Análise de distribuições
        distribution_summary = ""
        if len(numeric_cols) > 0:
            skew_analysis = []
            for col in numeric_cols[:5]:
                try:
                    skewness = df[col].skew()
                    if not np.isnan(skewness) and np.isfinite(skewness):
                        if abs(skewness) > 1:
                            skew_analysis.append(f"{col}: assimetria alta ({skewness:.2f})")
                        elif abs(skewness) > 0.5:
                            skew_analysis.append(f"{col}: assimetria moderada ({skewness:.2f})")
                except:
                    continue
            
            if skew_analysis:
                distribution_summary = f"Distribuições: {'; '.join(skew_analysis)}"
            else:
                distribution_summary = "Distribuições: aproximadamente normais"
        
        # Análise de qualidade detalhada
        missing_analysis = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                missing_analysis.append(f"{col}: {missing_pct:.1f}% faltante")
        
        # Calcular variabilidade
        variability_text = "N/A"
        if len(numeric_cols) > 0:
            try:
                cv_mean = (df[numeric_cols].std() / df[numeric_cols].mean()).mean()
                if not np.isnan(cv_mean) and np.isfinite(cv_mean):
                    variability_text = f"{cv_mean:.3f}"
            except:
                variability_text = "N/A"
        
        system_context = """Você é um Senior Data Scientist com 15+ anos de experiência em análise exploratória de dados. 
        Sua especialidade é identificar padrões, anomalias e oportunidades de insight em datasets complexos.
        Forneça análises TÉCNICAS, ESPECÍFICAS e QUANTITATIVAS. Evite respostas genéricas."""
        
        # Preparar variáveis para o prompt
        stats_text = "Sem colunas numéricas para análise estatística"
        if len(numeric_cols) > 0:
            stats_text = df.describe().to_string()
        
        missing_text = "Dataset completo"
        if missing_analysis:
            missing_text = '; '.join(missing_analysis[:10])
        
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
        {stats_text}

        🔗 ANÁLISE DE CORRELAÇÕES:
        {correlation_summary}

        📐 ANÁLISE DE DISTRIBUIÇÕES:
        {distribution_summary}

        🔍 QUALIDADE DOS DADOS:
        - Valores faltantes: {missing_text}
        - Duplicatas: {df.duplicated().sum():,} registros
        - Variabilidade: CV médio = {variability_text}

        FORNEÇA UMA ANÁLISE ESTRUTURADA E TÉCNICA:

        ## IDENTIFICAÇÃO DO DOMÍNIO
        Com base nos nomes das colunas, distribuições e padrões, identifique especificamente o tipo de dataset.

        ## AVALIAÇÃO TÉCNICA DE QUALIDADE
        Avalie objetivamente: completude, consistência, outliers potenciais, balanceamento.

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
        
        # Parsing da resposta
        analysis_result = self._parse_detailed_analysis(response, df)
        
        self.dataset_context = analysis_result
        self._add_to_gemini_memory("initial_analysis", analysis_result)
        
        return analysis_result
    
    def _parse_detailed_analysis(self, response: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse robusto da análise detalhada"""
        try:
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
            return {
                "dataset_type": self._generate_fallback_domain_analysis(df),
                "data_quality": self._generate_fallback_quality_analysis(df),
                "key_characteristics": self._generate_fallback_characteristics(df),
                "recommended_analyses": self._generate_fallback_recommendations(df),
                "potential_insights": self._generate_fallback_insights(df),
                "full_response": response
            }
    
    def process_user_query(self, user_query: str, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Processa query do usuário usando Gemini"""
        
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
        
        # FASE 4: CRIAR VISUALIZAÇÃO SE NECESSÁRIO
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
        Crie um plano de análise em formato JSON:
        {{
            "analysis_type": "[tipo_da_analise]",
            "columns_to_use": ["coluna1", "coluna2"],
            "requires_visualization": [true/false],
            "visualization_type": "[tipo_do_grafico]"
        }}

        **Tipos de Análise Válidos:**
        - `descriptive_statistics`: estatísticas básicas
        - `correlation_matrix`: correlações
        - `distribution_plot`: distribuição de variável
        - `outlier_detection`: outliers
        - `clustering`: agrupamentos
        - `frequency_analysis`: valores frequentes
        - `temporal_analysis`: padrões temporais
        - `balance_analysis`: balanceamento
        - `general_query`: perguntas gerais

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
        """Executa o plano de análise"""
        
        analysis_type = plan.get("analysis_type", "general_query")
        columns = plan.get("columns_to_use", [])
        
        system_context = """Você é um especialista em Python para análise de dados. Gere código Python para realizar a análise solicitada."""
        
        prompt = f"""
        **Plano de Análise:**
        - Tipo: {analysis_type}
        - Colunas: {columns}

        **Sua Tarefa:**
        Gere código Python completo para esta análise no DataFrame `df`:
        1. Use pandas, numpy, sklearn
        2. Imprima os resultados em formato claro
        3. Não crie gráficos, apenas análise textual

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
        """Gera a resposta final"""
        
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
        Escreva uma resposta clara:
        1. Explique o que foi analisado
        2. Resumir os principais resultados
        3. Forneça insights práticos
        4. Use linguagem acessível
        
        **Sua Resposta:**
        """
        
        return self._call_gemini(prompt, system_context)

    def _create_visualization_if_needed(self, plan: Dict, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        """Gera visualização se necessário"""
        
        if not plan.get("requires_visualization"): 
            return None
        
        visualization_type = plan.get("visualization_type")
        columns = plan.get("columns_to_use")
        
        system_context = """Você é um especialista em visualização de dados com Python."""
        
        prompt = f"""
        **Plano de Visualização:**
        - Tipo de Gráfico: {visualization_type}
        - Colunas: {columns}

        **Sua Tarefa:**
        Gere código Python para criar este gráfico:

        1. Criar: `generated_fig, ax = plt.subplots(figsize=(10, 6))`
        2. Gerar o gráfico no eixo `ax`
        3. Incluir título e rótulos
        4. Terminar com `plt.tight_layout()`
        5. A figura deve estar na variável `generated_fig`

        **Seu Código Python:**
        """
        
        code_to_execute = self._call_gemini(prompt, system_context)
        
        code_to_execute = code_to_execute.strip().replace("```python", "").replace("```", "").strip()
        
        try:
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
            
            exec(code_to_execute, exec_globals)
            
            generated_fig = exec_globals.get('generated_fig')
            
            if generated_fig is None:
                return {"status": "error", "error_message": "Figura não foi criada corretamente", "code_executed": code_to_execute}
            
            return {"status": "success", "figure": generated_fig, "code_executed": code_to_execute}
            
        except Exception as e:
            st.error(f"Erro ao gerar visualização: {e}")
            st.code(code_to_execute, language="python")
            return {"status": "error", "error_message": str(e), "code_executed": code_to_execute}

    # Métodos auxiliares
    def _extract_robust_section(self, text: str, start_markers: List[str], end_markers: List[str]) -> str:
        """Extração robusta de seções"""
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
                    section = section.replace("##", "").replace("#", "").strip()
                    
                    if len(section) > 10:
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
                if len(item) > 15:
                    items.append(item)
        
        return items[:5]
    
    def _generate_fallback_domain_analysis(self, df: pd.DataFrame) -> str:
        """Gera análise de domínio robusta"""
        cols_lower = [col.lower() for col in df.columns]
        
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
        
        if completeness > 95 and duplicates == 0:
            quality_score = "EXCELENTE"
        elif completeness > 90:
            quality_score = "BOA"
        elif completeness > 80:
            quality_score = "MODERADA"
        else:
            quality_score = "BAIXA"
        
        return f"Qualidade {quality_score}: {completeness:.1f}% completo, {duplicates:,} duplicatas, {df.shape[0]:,} registros válidos para análise"
    
    def _generate_fallback_characteristics(self, df: pd.DataFrame) -> List[str]:
        """Gera características robustas"""
        characteristics = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            skew_analysis = df[numeric_cols].skew()
            highly_skewed = sum(abs(skew_analysis) > 1)
            characteristics.append(f"Distribuições: {highly_skewed} de {len(numeric_cols)} variáveis numéricas apresentam assimetria alta")
            
            try:
                cv_values = df[numeric_cols].std() / df[numeric_cols].mean()
                cv_mean = cv_values.mean()
                if not np.isnan(cv_mean) and np.isfinite(cv_mean):
                    if cv_mean > 1:
                        characteristics.append(f"Variabilidade alta: coeficiente de variação médio de {cv_mean:.2f} indica dados heterogêneos")
                    else:
                        characteristics.append(f"Variabilidade moderada: coeficiente de variação médio de {cv_mean:.2f}")
                else:
                    characteristics.append("Variabilidade: não calculável devido a valores zero ou inválidos")
            except:
                characteristics.append("Análise de variabilidade: dados requerem pré-processamento")
            
            if len(numeric_cols) >= 2:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    high_corr_count = 0
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7 and not np.isnan(corr_matrix.iloc[i, j]):
                                high_corr_count += 1
                    characteristics.append(f"Estrutura de correlação: {high_corr_count} pares de variáveis altamente correlacionadas")
                except:
                    characteristics.append("Estrutura de correlação: análise requer limpeza de dados")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            high_cardinality = sum(df[col].nunique() > 50 for col in categorical_cols)
            if high_cardinality > 0:
                characteristics.append(f"Cardinalidade: {high_cardinality} variáveis categóricas com alta diversidade (>50 valores únicos)")
        
        if not characteristics:
            characteristics = ["Dataset com estrutura padrão para análise exploratória"]
        
        return characteristics[:3]
    
    def _generate_fallback_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Gera recomendações robustas"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            recommendations.append(f"Análise de correlação entre {len(numeric_cols)} variáveis numéricas para identificar relacionamentos lineares")
        
        if len(numeric_cols) > 0:
            recommendations.append("Detecção de outliers multivariada usando Isolation Forest para identificar anomalias")
        
        if df.shape[0] > 1000:
            recommendations.append("Clustering hierárquico ou K-means para segmentação e identificação de padrões latentes")
        
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            recommendations.append(f"Análise de séries temporais na coluna {time_cols[0]} para identificar tendências e sazonalidade")
        
        if not recommendations:
            recommendations = ["Análise exploratória sistemática das variáveis principais"]
        
        return recommendations[:3]
    
    def _generate_fallback_insights(self, df: pd.DataFrame) -> List[str]:
        """Gera insights potenciais robustos"""
        insights = []
        
        if df.shape[0] > 10000:
            insights.append(f"Grande volume de dados ({df.shape[0]:,} registros) permite análises estatísticas robustas e modelagem preditiva")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:2]:
            value_counts = df[col].value_counts()
            if len(value_counts) == 2:
                balance_ratio = value_counts.min() / value_counts.max()
                if balance_ratio < 0.1:
                    insights.append(f"Forte desbalanceamento na variável {col} ({balance_ratio:.2f}) sugere necessidade de técnicas de balanceamento")
        
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if len(missing_cols) > df.shape[1] * 0.3:
            insights.append(f"Padrão de dados faltantes em {len(missing_cols)} variáveis pode indicar processo de coleta estruturado")
        
        if not insights:
            insights = ["Potencial para descoberta de padrões não óbvios através de análise multivariada"]
        
        return insights[:3]

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
        if not hasattr(st.session_state, 'gemini_memory') or not st.session_state.gemini_memory:
            return summary_text + "Nenhuma análise realizada ainda."

        for i, entry in enumerate(st.session_state.gemini_memory):
            try:
                summary_text += f"\n--- **Análise {i+1}: {entry['type'].replace('_', ' ').title()}** ---\n"
                summary_text += f"**Timestamp:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                if entry["type"] == "initial_analysis":
                    full_response = entry["data"].get("full_response", "Análise não disponível")
                    summary_text += f"**Análise Inicial do Dataset:**\n{full_response}\n"
                elif entry["type"] == "user_query":
                    query = entry["data"].get("query", "Pergunta não disponível")
                    plan = entry["data"].get("plan", {})
                    results = entry["data"].get("results", {})
                    response = entry["data"].get("response", "Resposta não disponível")
                    
                    summary_text += f"**Pergunta do Usuário:** {query}\n"
                    summary_text += f"**Plano de Análise:** {json.dumps(plan, indent=2)}\n"
                    
                    text_output = results.get("text_output", "Resultados não disponíveis")
                    summary_text += f"**Resultados da Execução:**\n```\n{text_output}\n```\n"
                    summary_text += f"**Resposta Final do Gemini:**\n{response}\n"
            except Exception as e:
                summary_text += f"**Erro ao processar análise {i+1}:** {str(e)}\n"
            
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

# Funções auxiliares para compatibilidade
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
    if not PDF_AVAILABLE:
        raise ImportError("PdfPages não está disponível. Instale matplotlib com suporte a PDF.")
    
    pdf_buffer = io.BytesIO()
    
    try:
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

            # Páginas subsequentes: Análises realizadas
            if hasattr(st.session_state, 'gemini_memory') and st.session_state.gemini_memory:
                for i, entry in enumerate(st.session_state.gemini_memory):
                    try:
                        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                        ax = fig.add_subplot(111)
                        
                        title = f"Análise {i+1}: {entry['type'].replace('_', ' ').title()}"
                        ax.text(0.05, 0.95, title, ha="left", va="top", fontsize=14, fontweight="bold")
                        
                        content = ""
                        if entry["type"] == "initial_analysis":
                            data = entry.get("data", {})
                            content = data.get("full_response", "Análise inicial não disponível")
                        elif entry["type"] == "user_query":
                            data = entry.get("data", {})
                            query = data.get("query", "Pergunta não disponível")
                            response = data.get("response", "Resposta não disponível")
                            
                            content = f"Pergunta: {query}\n\nResposta Final:\n{response}"
                        
                        content = content[:1500] if content else "Conteúdo não disponível"
                        
                        ax.text(0.05, 0.90, content, ha="left", va="top", fontsize=8)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis("off")
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

                        # Adicionar gráficos se existirem
                        if (entry["type"] == "user_query" and 
                            entry.get("data", {}).get("visualization") and 
                            entry["data"]["visualization"].get("status") == "success" and
                            entry["data"]["visualization"].get("figure")):
                            
                            try:
                                fig_vis = entry["data"]["visualization"]["figure"]
                                if fig_vis:
                                    pdf.savefig(fig_vis, bbox_inches="tight")
                                    plt.close(fig_vis)
                            except Exception:
                                continue
                                
                    except Exception as e:
                        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                        ax = fig.add_subplot(111)
                        ax.text(0.05, 0.95, f"Erro na Análise {i+1}", ha="left", va="top", fontsize=14, fontweight="bold")
                        ax.text(0.05, 0.90, f"Erro: {str(e)}", ha="left", va="top", fontsize=10)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis("off")
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
            else:
                fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Nenhuma análise realizada ainda.\nFaça algumas perguntas ao agente primeiro.", 
                       ha="center", va="center", fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erro ao gerar PDF: {str(e)}")
        return generate_text_report_fallback(df, agent)

def generate_text_report_fallback(df, agent: GeminiAgent):
    """Fallback: gera relatório em texto quando PDF falha"""
    try:
        report_content = agent.get_full_memory_summary()
        
        dataset_info = get_dataset_info(df)
        full_report = f"""
RELATÓRIO COMPLETO DE ANÁLISE DE DADOS
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

{dataset_info}

{report_content}
"""
        
        return full_report.encode('utf-8')
        
    except Exception as e:
        simple_report = f"""
RELATÓRIO DE ANÁLISE - ERRO NA GERAÇÃO
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas

Erro na geração do relatório: {str(e)}

Para obter um relatório completo, tente:
1. Verificar as dependências do matplotlib
2. Executar algumas análises primeiro
3. Verificar a configuração do Gemini
"""
        return simple_report.encode('utf-8')

def initialize_gemini_agent():
    """Inicializa o agente Gemini no Streamlit"""
    if 'gemini_agent' not in st.session_state:
        st.session_state.gemini_agent = GeminiAgent()
    
    return st.session_state.gemini_agent

def get_adaptive_suggestions(df):
    """Função de compatibilidade - usa o agente Gemini se disponível"""
    if 'gemini_agent' in st.session_state and st.session_state.gemini_agent.model:
        return st.session_state.gemini_agent.generate_smart_suggestions(df)
    else:
        return [
            "Mostre estatísticas descritivas das colunas numéricas",
            "Analise correlações entre as variáveis",
            "Detecte outliers nos dados",
            "Faça clustering automático",
            "Mostre a distribuição da coluna principal",
            "Qual a memória do agente?"
        ]

