# 🤖 Agente Autônomo de Análise de Dados

Este projeto é um **agente inteligente de análise de dados** desenvolvido em **Python + Streamlit**, capaz de realizar análises automáticas em qualquer dataset (CSV), com **memória adaptativa** e **interpretação em linguagem natural**.

## 🚀 Funcionalidades

* 📊 **Estatísticas descritivas automáticas**
* 🔗 **Mapa de correlação entre variáveis**
* ⏰ **Análise de padrões temporais**
* 🎯 **Clustering (agrupamento automático)**
* ⚠️ **Detecção de outliers (Isolation Forest)**
* 📋 **Sugestões de perguntas adaptativas**
* 🧠 **Memória persistente das análises realizadas**

## 📂 Estrutura do projeto

```
agente-analise-dados/
│
├── app.py             # Interface principal (Streamlit)
├── utils.py           # Funções de análise e memória
├── requirements.txt   # Dependências
└── README.md          # Documentação
```
## 📂 Dataset

Este projeto utiliza o dataset público **Credit Card Fraud Detection**, disponível no Kaggle:

🔗 [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

⚠️ O arquivo `creditcard.csv` tem aproximadamente **147 MB**, e por isso não está incluído neste repositório.

**Instruções para uso:**
1. Baixe o arquivo `creditcard.csv` do Kaggle.
2. Crie uma pasta chamada `data/` na raiz do projeto.
3. Coloque o arquivo dentro dela

## 🛠️ Instalação local

Clone o repositório:

```bash
git clone https://github.com/anamanuellar/agente-analise-dados.git
cd agente-analise-dados
```

Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

Acesse no navegador: [http://localhost:8501](http://localhost:8501)

## ☁️ Deploy no Streamlit Cloud

1. Suba este repositório no GitHub.
2. Vá em [Streamlit Cloud](https://share.streamlit.io).
3. Conecte sua conta ao GitHub.
4. Escolha este repositório e defina `app.py` como arquivo principal.
5. Clique em **Deploy** 🚀.

O Streamlit Cloud gerará um link público para compartilhar sua aplicação.

## 📄 Exemplos de perguntas

Você pode interagir em **linguagem natural**. Alguns exemplos:

* *"Mostre estatísticas descritivas"*
* *"Mostre correlações entre variáveis"*
* *"Qual a distribuição da coluna Amount?"*
* *"A frequência de fraudes varia no tempo?"*
* *"Faça clustering automático dos dados"*
* *"Detecte outliers na coluna Time"*
* *"Qual sua memória de análises?"*

---

👩‍💻 **Autora**: Ana Manuella Ribeiro
📅 **Versão**: Setembro 2025
