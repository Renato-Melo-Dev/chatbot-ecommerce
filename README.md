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
Abra o **VS Code** → *File → Open Folder* → selecione a pasta (`chatbot-ecommerce/`).  

### 2. Criar e ativar ambiente virtual  
No terminal integrado (Ctrl+`):  

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar no Linux/Mac
source .venv/bin/activate

# Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1

```
### 2. Criar e ativar ambiente virtual
pip install -r requirements.txt

### 4. Rodar o Streamlit
streamlit run app/main_app.py

### 5. Estrutura de código

- Front-end: app/main_app.py (UI em Streamlit)

- Back-end: core/ (dados, SQL, modelos, integrações)

- Banco: ecommerce.db (SQLite, recriado a cada execução)

- Notebooks: notebooks/ (exploração de dados e ML)

### 📂 Estrutura de pastas
chatbot-ecommerce/
├─ app/            # Interface Streamlit
│   └─ main_app.py
├─ core/           # SQL, modelos e funções auxiliares
│   └─ sql/        # Scripts SQL executados na inicialização
├─ data/           # Dados brutos e tratados (CSV, SQLite)
├─ models/         # Modelos salvos (.pkl)
├─ notebooks/      # Notebooks de exploração (EDA e ML)
├─ tests/          # Testes unitários e de integração
├─ docs/           # Documentação (PMC, arquitetura, LGPD etc.)
├─ requirements.txt
└─ README.md

