# 📊 E-commerce Dashboard MVP  

MVP educacional para **análise de vendas** e **previsão de preços** em datasets de e-commerce.  
O projeto utiliza **Streamlit** para a interface, **SQLite** para persistência de dados e **Machine Learning (Linear Regression)** para previsão.  

---

## 📖 Documentação  
A pasta `docs/` pode conter:  
- **PMC** – Planejamento e Metodologia de Coleta  
- **Arquitetura** – Estrutura do sistema  
- **Modelagem de Dados** – Tabelas `SOR`, `SOT` e `SPEC`  
- **Governança LGPD/DAMA** – Boas práticas de dados  
- **Testes** – Estratégias de validação  
- **Deploy** – Como publicar o app  

---

## 🖥️ Como rodar o projeto no Visual Studio Code  

### 1. Abrir o projeto  
Abra o **VS Code → File → Open Folder** e selecione a pasta do projeto (`chatbot-ecommerce/`).  

### 2. Criar e ativar ambiente virtual  
No terminal integrado (Ctrl+`):

```bash
# Criar ambiente virtual
python -m venv .venv
```
```bash
# Ativar no Linux/Mac
source .venv/bin/activate
```
```bash
# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
```bash
### 3. Instalar dependências
pip install -r requirements.txt
```
```bash
4. Rodar o Streamlit
streamlit run app/main.py
```
### 5. Como usar o app

Faça upload de um CSV com dados de vendas.

Na Sidebar, clique em Treinar modelo para gerar o modelo de previsão.

Clique em Carregar modelo e prever para gerar as previsões.

Use as abas para visualizar Resultados do Treino, Predições ou conversar com o Chatbot sobre métricas e features.

### 📂 Estrutura de pastas
chatbot-ecommerce/
├─ app/                # Interface Streamlit
│   └─ main.py
├─ core/               # SQL, modelos e funções auxiliares
│   └─ chatbot/        # Regras do chatbot
├─ data/               # Dados brutos e scripts SQL
├─ models/             # Scripts de treino e predição
├─ models_store/       # Modelos treinados (.pkl)
├─ notebooks/          # Notebooks de exploração (EDA e ML)
├─ docs/               # Documentação (PMC, arquitetura, LGPD etc.)
├─ requirements.txt    # Dependências Python
└─ README.md           # Este arquivo