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
    """Agente que usa Google Gemini como cérebro do sistema"""

    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.model = None  # Inicializar após configurar API Key
        self.conversation_history = []
        self.dataset_context = {}

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4000,
        }

    def configure_gemini(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        if not self.model:
            return self._fallback_response("Gemini não configurado. Por favor, forneça a API Key.")

        try:
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            response = self.model.generate_content(full_prompt)

            texts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    texts.append(part.text)
                elif hasattr(part, "executable_code") and part.executable_code.code:
                    code_text = part.executable_code.code.lstrip("\n")
                    texts.append(f"``````")
            return "\n".join(texts)

        except Exception as e:
            st.error(f"Erro no Gemini: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
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
        system_context = """Você é um especialista sênior em análise de dados com PhD em Estatística e vasta experiência em Business Intelligence.""" 
        prompt = f"""
        Analise este dataset CSV e forneça uma análise estruturada seguindo formato técnico e prático.

        INFORMAÇÕES:
        Linhas: {df.shape[0]:,}
        Colunas: {df.shape[1]}
        Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
        Colunas numéricas: {len(df.select_dtypes(include=[np.number]).columns)}
        Colunas categóricas: {len(df.select_dtypes(include=["object"]).columns)}
        Dados faltantes: {df.isnull().sum().sum():,}
        Nomes colunas: {list(df.columns)}
        Amostra 3 primeiras linhas:
        {df.head(3).to_string()}
        """
        response = self._call_gemini(prompt, system_context)
        self.dataset_context = {"full_response": response}
        return {"full_response": response}

    def process_user_query(self, user_query: str, df: pd.DataFrame) -> Tuple[str, Dict]:
        self.conversation_history.append({"role": "user", "content": user_query, "timestamp": datetime.now()})

        analysis_plan = self._gemini_create_analysis_plan(user_query, df)
        analysis_results = self._execute_gemini_analysis_plan(analysis_plan, df)
        final_response = self._gemini_generate_final_response(user_query, analysis_plan, analysis_results, df)
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
        system_context = "Você é um agente especialista em análise de dados. Crie um JSON detalhado do plano de análise."
        prompt = f"""
        Pergunta: "{user_query}"
        Colunas disponiveis: {list(df.columns)}
        Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)}
        Colunas categóricas: {list(df.select_dtypes(include=['object']).columns)}

        Retorne somente JSON com os campos:
        {{"analysis_type": "", "columns_to_use": [], "requires_visualization": false, "visualization_type": ""}}
        """
        response = self._call_gemini(prompt, system_context)
        try:
            cleaned = response.strip().replace("``````", "").strip()
            plan = json.loads(cleaned)
        except Exception:
            plan = {"analysis_type": "general_query", "columns_to_use": [], "requires_visualization": False, "visualization_type": ""}
        return plan

    def _execute_gemini_analysis_plan(self, plan: Dict, df: pd.DataFrame) -> Dict:
        analysis_type = plan.get("analysis_type", "general_query")
        columns = plan.get("columns_to_use", [])

        system_context = "Especialista Python para análise, gere código para análise solicitada e imprima texto."

        prompt = f"""
        Plano: {analysis_type}
        Colunas: {columns}

        Gere código Python para análise, imprima resultados textuais claros.
        """

        code = self._call_gemini(prompt, system_context)
        code_cleaned = code.strip().replace("``````", "").strip()

        try:
            output_buffer = io.StringIO()
            exec_globals = {
                "df": df, "pd": pd, "np": np, "plt": plt, "sns": sns,
                "StandardScaler": StandardScaler,
                "IsolationForest": IsolationForest,
                "KMeans": KMeans,
                "silhouette_score": silhouette_score,
                "print": lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)
            }
            exec(code_cleaned, exec_globals)
            output = output_buffer.getvalue()
            return {"status": "success", "text_output": output, "code_executed": code_cleaned}
        except Exception as e:
            return {"status": "error", "error_message": str(e), "code_executed": code_cleaned}

    def _gemini_generate_final_response(self, user_query: str, plan: Dict, results: Dict, df: pd.DataFrame) -> str:
        system_context = "Especialista em análise explicando resultados para cliente de negócio."
        prompt = f"""
        Pergunta do Cliente: {user_query}
        Análise: {plan.get('analysis_type')}
        Colunas: {plan.get('columns_to_use')}
        Resultados:
        ```
        {results.get('text_output')}
        ```
        Gere uma resposta clara com insights e recomendações práticas.
        """
        return self._call_gemini(prompt, system_context)

    def _create_visualization_if_needed(self, plan: Dict, df: pd.DataFrame, analysis_results: Dict) -> Dict:
        if not plan.get("requires_visualization"):
            return None

        visualization_type = plan.get("visualization_type")
        columns = plan.get("columns_to_use")

        system_context = (
            "Você é um especialista em visualização de dados com Python. "
            "Gere código para criar figura. Não use plt.show() ou fig.show(). "
            "Use plt.tight_layout() para evitar sobreposição."
        )

        prompt = f"""
        Plano de Visualização:
        Tipo: {visualization_type}
        Colunas: {columns}
        Resultados da análise textual: {analysis_results.get('text_output', 'N/A')}

        Gere o código Python para o gráfico solicitado, retornando a figura (fig) na última linha.
        """

        code_to_execute = self._call_gemini(prompt, system_context)
        code_cleaned = re.sub(r"``````").strip()

        try:
            exec_globals = {
                "df": df, "pd": pd, "np": np, "plt": plt, "sns": sns,
                "StandardScaler": StandardScaler,
                "IsolationForest": IsolationForest,
                "KMeans": KMeans,
                "silhouette_score": silhouette_score,
            }
            fig = eval(code_cleaned, exec_globals)
            return {"status": "success", "figure": fig, "code_executed": code_cleaned}
        except Exception as e:
            st.error(f"Erro ao gerar visualização: {e}")
            st.code(code_cleaned, language="python")
            return {"status": "error", "error_message": str(e), "code_executed": code_cleaned}

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
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
        section = self._extract_section(text, start_marker, end_marker)
        return [item.strip().lstrip("- ") for item in section.split("\n") if item.strip().startswith("-")]

    def _add_to_gemini_memory(self, analysis_type: str, data: Dict):
        if "gemini_memory" not in st.session_state:
            st.session_state.gemini_memory = []

        st.session_state.gemini_memory.append({"type": analysis_type, "data": data, "timestamp": datetime.now()})

    def get_full_memory_summary(self) -> str:
        summary_text = "🤖 **Resumo Completo das Análises - Gemini Agent**\n\n"
        if not st.session_state.gemini_memory:
            return summary_text + "Nenhuma análise realizada ainda."

        for i, entry in enumerate(st.session_state.gemini_memory):
            summary_text += f"\n--- Análise {i+1}: {entry['type'].replace('_', ' ').title()} ---\n"
            summary_text += f"Timestamp: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            if entry["type"] == "initial_analysis":
                summary_text += entry["data"].get("full_response", "") + "\n"
            elif entry["type"] == "user_query":
                summary_text += f"Pergunta: {entry['data']['query']}\n"
                summary_text += f"Plano: {json.dumps(entry['data']['plan'], indent=2)}\n"
                summary_text += f"Resultados:\n``````\n"
                summary_text += f"Resposta Final:\n{entry['data']['response']}\n"
        return summary_text

    def generate_smart_suggestions(self, df: pd.DataFrame) -> List[str]:
        system_context = "Você é um especialista em análise de dados. Gere sugestões práticas de perguntas para o usuário."
        prompt = f"""
        Contexto Dataset:
        - Shape: {df.shape}
        - Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)[:8]}
        - Colunas categóricas: {list(df.select_dtypes(include=['object']).columns)[:8]}
        - Análises anteriores: {len(self.conversation_history)}

        Gere 5-6 sugestões claras de perguntas específicas úteis para o dataset.
        """

        response = self._call_gemini(prompt, system_context)
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(('-', '*', '•')):
                suggestion = line[1:].strip().strip('"')
                if suggestion:
                    suggestions.append(suggestion)
        return suggestions[:6]
