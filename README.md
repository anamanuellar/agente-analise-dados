# 🤖 Agente Autônomo de Análise de Dados com Google Gemini

Este projeto é um **agente inteligente de análise de dados** desenvolvido em **Python + Streamlit**, capaz de realizar análises automáticas em qualquer dataset (CSV), com **Google Gemini** como cérebro principal, **memória adaptativa** e **interpretação em linguagem natural**.

**🌐 Aplicação Online:** [https://agentanalysisdata.streamlit.app/](https://agentanalysisdata.streamlit.app/)

## 🧠 Powered by Google Gemini AI

**Framework Principal:** Google Gemini 2.0 Flash Experimental
- **Interpretação inteligente** de perguntas em linguagem natural
- **Geração automática** de código Python para análise
- **Memória conversacional** persistente
- **Análise contextual** profunda dos dados

## 🚀 Funcionalidades Avançadas

### 🔍 **Análise Automática Inteligente**
* 📊 **Identificação automática do domínio** dos dados
* 🧪 **Estatísticas descritivas** com interpretação contextual
* 🔗 **Matriz de correlação** com insights específicos
* ⏰ **Análise de padrões temporais** automatizada
* 🎯 **Clustering inteligente** (K-means otimizado)
* ⚠️ **Detecção de outliers** (Isolation Forest)
* 📋 **Avaliação de qualidade** dos dados

### 💬 **Chat Inteligente com IA**
* 🗣️ **Perguntas em linguagem natural** (português)
* 🤖 **Respostas contextuais** baseadas nos dados
* 📊 **Visualizações automáticas** quando relevantes
* 💡 **Sugestões inteligentes** personalizadas
* 🧠 **Memória persistente** de todas as interações

### 📈 **Visualizações Automáticas**
* 📊 **Gráficos gerados dinamicamente** pela IA
* 🔥 **Heatmaps de correlação** interativos
* 📈 **Séries temporais** com análise de tendências
* 🎯 **Clustering visual** com interpretação
* 📋 **Dashboards executivos** automáticos

### 📄 **Relatórios Profissionais**
* 📋 **Resumos executivos** gerados pela IA
* 📄 **Relatórios PDF** completos
* 🗃️ **Histórico de análises** acessível
* 💾 **Exportação** de resultados

## 📂 Estrutura do Projeto

```
agente-analise-dados/
│
├── app.py                 # Interface principal (Streamlit)
├── utils.py              # GeminiAgent + funções de análise
├── requirements.txt      # Dependências
├── README.md            # Documentação
└── data/               # Pasta para datasets (criar)
    └── creditcard.csv  # Dataset de exemplo (baixar)
```

## 🗃️ Dataset de Exemplo

Este projeto foi testado com o dataset público **Credit Card Fraud Detection** do Kaggle:

🔗 [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**⚠️ Dataset não incluído** (147 MB):
1. Baixe o arquivo `creditcard.csv` do Kaggle
2. Crie uma pasta `data/` na raiz do projeto  
3. Coloque o arquivo dentro dela

**Características do Dataset:**
- 📊 284.807 transações de cartão de crédito
- 🔒 30 features (V1-V28 são componentes PCA + Time + Amount + Class)
- ⚖️ Altamente desbalanceado (0,17% fraudes)
- 🎯 Ideal para demonstrar capacidades do agente

## 🛠️ Instalação e Uso

### **Pré-requisitos:**
- Python 3.8+
- API Key do Google Gemini (gratuita)

### **1. Clone o repositório:**

```bash
git clone https://github.com/anamanuellar/agente-analise-dados.git
cd agente-analise-dados
```

### **2. Instale as dependências:**

```bash
pip install -r requirements.txt
```

### **3. Execute a aplicação:**

```bash
streamlit run app.py
```

### **4. Configure o Gemini:**
1. Acesse: https://aistudio.google.com/app/apikey
2. Faça login com conta Google
3. Clique "Create API Key" 
4. Cole a chave na barra lateral da aplicação

### **5. Carregue seus dados:**
- Upload de arquivo CSV
- Aguarde análise automática
- Comece a fazer perguntas!

## ☁️ Deploy no Streamlit Cloud

1. **Fork** este repositório no GitHub
2. Acesse [Streamlit Cloud](https://share.streamlit.io)
3. Conecte sua conta GitHub
4. Selecione este repositório
5. Configure `app.py` como arquivo principal
6. **Deploy** 🚀

## 🎯 Exemplos de Uso

### **💬 Perguntas que o Agente Entende:**

**Análises Básicas:**
- *"Mostre estatísticas descritivas completas"*
- *"Analise a qualidade dos dados"*
- *"Quais são as correlações mais importantes?"*

**Análises Avançadas:**
- *"Detecte outliers e explique o que encontrou"*
- *"Faça clustering automático e visualize os grupos"*
- *"Analise padrões temporais nos dados"*
- *"Qual a distribuição da variável Amount?"*

**Insights de Negócio:**
- *"Quais são as principais conclusões sobre fraudes?"*
- *"Como posso melhorar a detecção de anomalias?"*
- *"Que estratégias você recomenda baseado nos dados?"*

**Memória e Histórico:**
- *"Qual sua memória de análises anteriores?"*
- *"Resume tudo que analisamos até agora"*
- *"Gere um relatório executivo completo"*

### **📊 Resultados Demonstrados:**

**✅ Análise Inicial Automática:**
- Identificação: "Dataset de Detecção de Fraude em Transações Financeiras"
- Qualidade: "99.7% completo, 1.081 duplicatas, forte desbalanceamento"
- Características: Variáveis PCA, componente temporal, classificação binária

**✅ Correlações Inteligentes:**
- Heatmap automático de 30 variáveis
- Identificação de correlações significativas
- Insights sobre relacionamentos entre features

**✅ Clustering Automático:**
- 3 clusters otimizados (Silhouette Score: 0.68)
- Cluster 2 concentra 89% das fraudes
- Segmentação por valor e horário

**✅ Detecção de Outliers:**
- 1.456 outliers identificados (0.5% do dataset)
- Concentração em transações de alto valor
- 23% correlacionados com fraudes

## 🔧 Arquitetura Técnica

### **🧠 GeminiAgent Class:**
```python
class GeminiAgent:
    def analyze_dataset_initially()     # Análise automática inicial
    def process_user_query()           # Processamento de perguntas
    def _gemini_create_analysis_plan()  # Planejamento inteligente
    def _execute_gemini_analysis_plan() # Execução de código
    def get_full_memory_summary()       # Resumos executivos
```

### **🔄 Fluxo de Processamento (4 Fases):**
1. **Interpretação:** Gemini converte pergunta em plano estruturado
2. **Execução:** Código Python gerado e executado automaticamente  
3. **Resposta:** Gemini interpreta resultados e gera insights
4. **Visualização:** Criação automática de gráficos quando relevante

### **🗃️ Sistema de Memória:**
- Todas as interações armazenadas
- Referência cruzada entre análises
- Relatórios consolidados automáticos
- Exportação em PDF profissional

## 📋 Dependências

```
streamlit>=1.39.0
pandas>=2.2.2
numpy>=1.26.4
matplotlib>=3.9.2
seaborn>=0.13.2
scikit-learn>=1.5.1
plotly>=5.24.1
google-generativeai>=0.8.5
```

## 🎯 Casos de Uso

### **🏦 Análise Financeira:**
- Detecção de fraudes em tempo real
- Análise de padrões de transação
- Segmentação de clientes
- Avaliação de risco

### **📊 Análise de Negócios:**
- KPIs automatizados
- Insights de vendas
- Análise de comportamento
- Relatórios executivos

### **🔬 Pesquisa e Educação:**
- Análise exploratória automática
- Ensino de ciência de dados
- Prototipagem rápida
- Validação de hipóteses

### **🏥 Análise de Dados Médicos:**
- Estudos epidemiológicos
- Análise de eficácia
- Detecção de anomalias
- Relatórios clínicos

## 🚀 Próximas Funcionalidades

- [ ] **Suporte a múltiplos formatos** (Excel, JSON, SQL)
- [ ] **Integração com APIs** de dados
- [ ] **Modelos de ML automáticos** (AutoML)
- [ ] **Dashboard executivo** em tempo real
- [ ] **Colaboração multi-usuário**
- [ ] **Exportação para PowerBI/Tableau**

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature
3. **Commit** suas mudanças
4. **Push** para a branch
5. **Abra** um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 👩‍💻 Autora

**Ana Manuella Ribeiro**
- 📧 Email: [ribeiro.anamanuella@gmail.com]
- 💼 LinkedIn: [Clique aqui](https://www.linkedin.com/in/manu-ribeiro-dev/)
- 🌐 GitHub: [@anamanuellar](https://github.com/anamanuellar)

---

## 🔗 Links Úteis

- 🧠 **Google Gemini API:** https://aistudio.google.com/app/apikey
- 📊 **Dataset Exemplo:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- 🚀 **Streamlit Cloud:** https://share.streamlit.io
- 📚 **Documentação Streamlit:** https://docs.streamlit.io

---

<div align="center">

🤖 **Agente Autônomo de Análise de Dados** 🤖

🧠 *Powered by Google Gemini* | 🎯 *IA que Entende Dados* | 🚀 *Análise Inteligente*

---

⭐ **Se este projeto foi útil, deixe uma estrela!** ⭐

</div>
