# HybridGeminiAgent - VERSÃO COMPLETA CORRIGIDA
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
from matplotlib.backends.backend_pdf import PdfPages

class HybridGeminiAgent:
    """
    Agente Híbrido COMPLETO que combina:
    - LLM (Gemini) para interpretação e insights
    - Funções robustas para execução das análises
    - Interface completa da versão 1
    """
    
    def __init__(self, model_name="gemini-2.5-flash"):  
        self.model_name = model_name
        self.model = None
        self.conversation_history = []
        self.dataset_context = {}
        self.api_key = None
        
        # Configurações do Gemini
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2000,
        }
        
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
    
    def configure_gemini(self, api_key=None):
        """Configura o Gemini com a API Key - VERSÃO CORRIGIDA"""
        try:
            # 1. Primeiro, tentar obter a API key de diferentes fontes
            if api_key:
                self.api_key = api_key
            elif hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                self.api_key = st.secrets['GEMINI_API_KEY']
            elif hasattr(st, 'secrets') and 'gemini' in st.secrets and 'api_key' in st.secrets['gemini']:
                self.api_key = st.secrets['gemini']['api_key']
            else:
                st.error("❌ API Key do Gemini não encontrada! Verifique seu arquivo secrets.toml")
                return False
            
            # 2. Configurar o Gemini
            genai.configure(api_key=self.api_key)
            
            # 3. Inicializar o modelo
            self.model = genai.GenerativeModel(
                self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )
            
            # 4. Testar a configuração
            test_response = self.model.generate_content("Teste de conexão. Responda apenas 'OK'")
            
            if test_response and test_response.text:
                st.success("✅ Gemini configurado com sucesso!")
                return True
            else:
                st.error("❌ Falha no teste de conexão com Gemini")
                return False
                
        except Exception as e:
            st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
            self.model = None
            return False
    
    def _call_gemini(self, prompt: str, system_context: str = "") -> str:
        """Chama Gemini de forma segura com fallback - VERSÃO CORRIGIDA"""
        
        # Verificar se o modelo está configurado
        if not self.model:
            # Tentar configurar automaticamente
            if not self.configure_gemini():
                return "❌ Gemini não configurado. Verifique sua API key em secrets.toml"
        
        try:
            # Construir prompt completo
            full_prompt = f"{system_context}\n\n{prompt}" if system_context else prompt
            
            # Fazer a chamada
            response = self.model.generate_content(full_prompt)
            
            # Verificar se há resposta válida
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "⚠️ Gemini retornou resposta vazia"
                
        except Exception as e:
            error_msg = str(e)
            
            # Tratar erros específicos
            if "API_KEY_INVALID" in error_msg:
                return "❌ API Key inválida. Verifique sua chave do Gemini"
            elif "QUOTA_EXCEEDED" in error_msg:
                return "⚠️ Cota da API excedida. Tente novamente mais tarde"
            elif "SAFETY" in error_msg:
                return "⚠️ Conteúdo bloqueado por questões de segurança"
            else:
                st.error(f"Erro no Gemini: {error_msg}")
                return f"❌ Erro na LLM: {error_msg}"
    
    def check_configuration(self):
        """Verifica se o Gemini está configurado corretamente"""
        if not self.model:
            return False, "Modelo não inicializado"
        
        if not self.api_key:
            return False, "API Key não encontrada"
        
        try:
            # Teste simples
            test_response = self.model.generate_content("Teste")
            return True, "Configuração OK"
        except Exception as e:
            return False, f"Erro na configuração: {str(e)}"
    
    def analyze_dataset_initially(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise inicial com Gemini + dados estruturados robustos"""
        
        # PARTE 1: Análise robusta (sempre funciona)
        basic_analysis = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_cols": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum(),
            "completeness": ((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100),
            "duplicates": df.duplicated().sum()
        }
        
        # PARTE 2: Inteligência com Gemini (com fallback)
        system_context = """Você é um especialista em análise de dados. 
        Analise o dataset e forneça insights contextuais em português."""
        
        prompt = f"""
        Analise este dataset CSV:
        
        Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas
        Colunas: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}
        Tipos: {basic_analysis['numeric_cols']} numéricas, {basic_analysis['categorical_cols']} categóricas
        Qualidade: {basic_analysis['completeness']:.1f}% completo, {basic_analysis['duplicates']} duplicatas
        
        Amostra:
        {df.head(3).to_string()}
        
        Forneça uma análise estruturada:
        1. TIPO DE DATASET: Identifique o domínio (fraude, vendas, etc.)
        2. CARACTERÍSTICAS PRINCIPAIS: 3 pontos importantes
        3. ANÁLISES RECOMENDADAS: 3 análises mais valiosas
        4. INSIGHTS POTENCIAIS: O que pode ser descoberto
        
        Seja conciso e focado em valor prático.
        """
        
        llm_response = self._call_gemini(prompt, system_context)
        
        # Combinar análises
        result = {
            **basic_analysis,
            "llm_analysis": llm_response,
            "dataset_type": self._extract_dataset_type(llm_response),
            "key_characteristics": self._extract_characteristics(llm_response),
            "recommended_analyses": self._extract_recommendations(llm_response),
            "timestamp": datetime.now().isoformat()
        }
        
        self.dataset_context = result
        return result
    
    def interpret_query_intelligently(self, user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        HÍBRIDO: LLM interpreta + mapeamento para funções robustas
        """
        
        # PARTE 1: Interpretação inteligente com Gemini
        system_context = """Você é um especialista em interpretação de queries sobre análise de dados.
        Classifique a pergunta e forneça contexto."""
        
        prompt = f"""
        Pergunta do usuário: "{user_query}"
        
        Dataset: {df.shape[0]} linhas, {df.shape[1]} colunas
        Colunas disponíveis: {list(df.columns)[:10]}
        
        Classifique a pergunta em UMA categoria:
        - descriptive: estatísticas básicas, resumos
        - correlation: correlações, relacionamentos
        - distribution: distribuições, histogramas
        - outliers: outliers, anomalias
        - clustering: agrupamentos, segmentação
        - temporal: padrões temporais, trends
        - frequency: valores frequentes, contagens
        - balance: balanceamento de classes
        - insights: conclusões, descobertas
        - memory: histórico, memória do agente
        
        Responda APENAS com a categoria e coluna(s) se aplicável:
        Formato: categoria|coluna1,coluna2
        Exemplo: correlation|Amount,Time
        """
        
        llm_classification = self._call_gemini(prompt, system_context)
        
        # PARTE 2: Mapeamento robusto (fallback para regras)
        query_lower = user_query.lower()
        
        # Parse da resposta da LLM
        try:
            if '|' in llm_classification:
                category, columns_str = llm_classification.split('|', 1)
                specific_columns = [col.strip() for col in columns_str.split(',') if col.strip()]
            else:
                category = llm_classification.strip()
                specific_columns = []
        except:
            category = "general"
            specific_columns = []
        
        # Fallback para regras se LLM falhar
        if category not in ['descriptive', 'correlation', 'distribution', 'outliers', 
                          'clustering', 'temporal', 'frequency', 'balance', 'insights', 'memory']:
            category = self._fallback_classification(query_lower)
            specific_columns = self._extract_columns_from_query(user_query, df)
        
        return {
            "category": category,
            "specific_columns": specific_columns,
            "original_query": user_query,
            "llm_interpretation": llm_classification,
            "confidence": "high" if '|' in llm_classification else "medium"
        }
    
    def generate_intelligent_response(self, query_result: Dict, analysis_result: Any, df: pd.DataFrame) -> str:
        """
        Gera resposta inteligente combinando resultados técnicos com insights da LLM
        """
        
        system_context = """Você é um consultor sênior em dados. 
        Transforme resultados técnicos em insights claros e actionables."""
        
        prompt = f"""
        Pergunta original: "{query_result['original_query']}"
        Tipo de análise: {query_result['category']}
        
        Resultados técnicos obtidos:
        {str(analysis_result)[:1000] if analysis_result else "Análise executada"}
        
        Dataset context: {self.dataset_context.get('dataset_type', 'Dataset genérico')}
        
        Gere uma resposta clara que:
        1. Responda diretamente à pergunta
        2. Explique os principais achados
        3. Forneça insights práticos
        4. Sugira próximos passos se relevante
        
        Use linguagem acessível e focada em valor de negócio.
        Máximo 300 palavras.
        """
        
        return self._call_gemini(prompt, system_context)
    
    def generate_smart_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Gera sugestões inteligentes baseadas no dataset"""
        
        system_context = """Gere 5 sugestões práticas de perguntas sobre análise de dados."""
        
        prompt = f"""
        Dataset: {self.dataset_context.get('dataset_type', 'Genérico')}
        Colunas: {list(df.columns)[:8]}
        Shape: {df.shape}
        
        Gere 5 perguntas específicas e práticas que um analista faria sobre este dataset.
        
        Formato: uma pergunta por linha, começando com "•"
        Exemplo:
        • Quais são as correlações mais fortes entre as variáveis numéricas?
        • Existem outliers significativos que precisam de investigação?
        """
        
        response = self._call_gemini(prompt, system_context)
        
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                suggestion = line[1:].strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:5]
    
    def generate_executive_conclusions(self, df: pd.DataFrame, memory: Dict) -> str:
        """Gera conclusões executivas com base na memória"""
        
        system_context = """Você é um CDO (Chief Data Officer) gerando conclusões executivas."""
        
        # Compilar histórico
        analyses_summary = []
        for conclusion in memory.get('conclusions', [])[-10:]:
            analyses_summary.append(f"- {conclusion.get('analysis_type', 'N/A')}: {conclusion.get('conclusion', 'N/A')[:100]}")
        
        prompt = f"""
        Dataset analisado: {df.shape[0]:,} registros, {df.shape[1]} variáveis
        Tipo: {self.dataset_context.get('dataset_type', 'Dataset genérico')}
        
        Análises realizadas:
        {chr(10).join(analyses_summary[:5])}
        
        Total de análises: {len(memory.get('conclusions', []))}
        
        Gere conclusões executivas estruturadas:
        
        ## 🎯 Resumo Executivo
        [Síntese em 2-3 frases]
        
        ## 🔍 Principais Descobertas  
        [3-4 descobertas mais importantes]
        
        ## 💼 Impacto no Negócio
        [Como isso afeta estratégia/operações]
        
        ## 🎯 Recomendações
        [3-4 ações específicas recomendadas]
        
        Seja conciso, objetivo e focado em valor executivo.
        """
        
        return self._call_gemini(prompt, system_context)
    
    # === MÉTODOS AUXILIARES ===
    
    def _extract_dataset_type(self, text: str) -> str:
        """Extrai tipo de dataset da resposta da LLM"""
        lines = text.lower().split('\n')
        for line in lines:
            if 'dataset' in line or 'tipo' in line:
                if 'fraude' in line:
                    return 'Detecção de Fraude'
                elif 'vendas' in line or 'sales' in line:
                    return 'Dados de Vendas'
                elif 'marketing' in line:
                    return 'Marketing Analytics'
                elif 'financeiro' in line or 'finance' in line:
                    return 'Dados Financeiros'
        return 'Dataset Genérico'
    
    def _extract_characteristics(self, text: str) -> List[str]:
        """Extrai características da resposta"""
        characteristics = []
        lines = text.split('\n')
        in_characteristics = False
        
        for line in lines:
            if 'característica' in line.lower() or 'principais' in line.lower():
                in_characteristics = True
                continue
            if in_characteristics and line.strip().startswith(('•', '-', '1.', '2.', '3.')):
                char = re.sub(r'^[•\-\d\.]\s*', '', line.strip())
                if char:
                    characteristics.append(char[:100])
                if len(characteristics) >= 3:
                    break
        
        return characteristics or ['Análise detalhada disponível']
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extrai recomendações da resposta"""
        recommendations = []
        lines = text.split('\n')
        in_recommendations = False
        
        for line in lines:
            if 'recomend' in line.lower() or 'análise' in line.lower():
                in_recommendations = True
                continue
            if in_recommendations and line.strip().startswith(('•', '-', '1.', '2.', '3.')):
                rec = re.sub(r'^[•\-\d\.]\s*', '', line.strip())
                if rec:
                    recommendations.append(rec[:100])
                if len(recommendations) >= 3:
                    break
        
        return recommendations or ['Análise exploratória', 'Detecção de padrões']
    
    def _fallback_classification(self, query_lower: str) -> str:
        """Classificação de fallback baseada em regras"""
        if any(word in query_lower for word in ['correlação', 'relaciona', 'correlation']):
            return 'correlation'
        elif any(word in query_lower for word in ['outlier', 'anomalia', 'atípico']):
            return 'outliers'
        elif any(word in query_lower for word in ['cluster', 'agrupamento', 'grupo']):
            return 'clustering'
        elif any(word in query_lower for word in ['distribuição', 'histograma']):
            return 'distribution'
        elif any(word in query_lower for word in ['frequente', 'comum', 'contagem']):
            return 'frequency'
        elif any(word in query_lower for word in ['tempo', 'temporal', 'trend']):
            return 'temporal'
        elif any(word in query_lower for word in ['balanceamento', 'balancear']):
            return 'balance'
        elif any(word in query_lower for word in ['conclusão', 'insight', 'descoberta']):
            return 'insights'
        elif any(word in query_lower for word in ['memória', 'histórico']):
            return 'memory'
        else:
            return 'descriptive'
    
    def _extract_columns_from_query(self, query: str, df: pd.DataFrame) -> List[str]:
        """Extrai nomes de colunas mencionadas na query"""
        columns_mentioned = []
        for col in df.columns:
            if col.lower() in query.lower():
                columns_mentioned.append(col)
        return columns_mentioned

# FUNÇÃO PARA INICIALIZAR O AGENTE NO STREAMLIT
def initialize_hybrid_agent():
    """Inicializa o agente híbrido no Streamlit"""
    
    # Verificar se já existe na sessão
    if 'hybrid_agent' not in st.session_state:
        st.session_state.hybrid_agent = HybridGeminiAgent()
    
    # Verificar configuração
    agent = st.session_state.hybrid_agent
    is_configured, status = agent.check_configuration()
    
    if not is_configured:
        st.warning(f"⚠️ Configurando Gemini... Status: {status}")
        
        # Tentar configurar
        success = agent.configure_gemini()
        
        if not success:
            st.error("""
            ❌ **Erro na configuração do Gemini**
            
            Verifique se:
            1. O arquivo `.streamlit/secrets.toml` existe
            2. Contém sua API key: `GEMINI_API_KEY = "sua_chave_aqui"`
            3. A chave é válida no Google AI Studio
            
            **Como obter a API key:**
            - Acesse: https://aistudio.google.com/app/apikey
            - Crie uma nova chave
            - Adicione ao secrets.toml
            """)
            return None
    
    return agent

# VERIFICAÇÃO DE SECRETS.TOML
def verificar_secrets():
    """Função para verificar se os secrets estão configurados"""
    try:
        # Método 1: GEMINI_API_KEY diretamente
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            key = st.secrets['GEMINI_API_KEY']
            if key and len(key) > 30:  # API keys do Google são longas
                return True, "✅ GEMINI_API_KEY encontrada"
            else:
                return False, "❌ GEMINI_API_KEY muito curta ou vazia"
        
        # Método 2: gemini.api_key
        elif hasattr(st, 'secrets') and 'gemini' in st.secrets:
            if 'api_key' in st.secrets['gemini']:
                key = st.secrets['gemini']['api_key']
                if key and len(key) > 30:
                    return True, "✅ gemini.api_key encontrada"
                else:
                    return False, "❌ gemini.api_key muito curta ou vazia"
        
        return False, "❌ Nenhuma API key encontrada em secrets.toml"
        
    except Exception as e:
        return False, f"❌ Erro ao verificar secrets: {str(e)}"

# Adicione isso temporariamente para verificar a configuração
if st.sidebar.button("🔧 Debug Gemini"):
    has_secrets, message = verificar_secrets()
    st.sidebar.write(message)
    
    if has_secrets:
        agent = st.session_state.hybrid_agent
        is_configured, status = agent.check_configuration()
        st.sidebar.write(f"Status: {status}")
        
        if is_configured:
            response = agent._call_gemini("Teste rápido")
            st.sidebar.success(f"Funcionando: {response[:50]}...")
