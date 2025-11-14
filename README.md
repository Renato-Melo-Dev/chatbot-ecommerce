# 📊 E-commerce Dashboard MVP  

MVP educacional para **análise de vendas** e **previsão de preços** em datasets de e-commerce.  
O projeto utiliza **Streamlit** para a interface, **SQLite** para persistência de dados e **Machine Learning (Linear Regression)** para previsão.  

> Observação: o banco SQLite (`ecommerce.db`) é criado automaticamente na pasta `data/` ao rodar o app, garantindo que os dados fiquem centralizados e organizados.  

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
### Criar ambiente virtual
python -m venv .venv
### Ativar no Linux/Mac
source .venv/bin/activate
### Ativar no Windows (PowerShell)
.venv\Scripts\Activate.ps1
### Instalar dependências
pip install -r requirements.txt
### Rodar o Streamlit
streamlit run app/main.py
```

📂 Estrutura de pastas

chatbot-ecommerce/
├─ app/                # Interface Streamlit
│   └─ main.py
├─ core/               # SQL, modelos e funções auxiliares
│   └─ chatbot/        # Regras do chatbot
├─ data/               # Dados brutos, scripts SQL e ecommerce.db
├─ models/             # Scripts de treino e predição
├─ models_store/       # Modelos treinados (.pkl)
├─ notebooks/          # Notebooks de exploração (EDA e ML)
├─ docs/               # Documentação (PMC, arquitetura, LGPD etc.)
├─ requirements.txt    # Dependências Python
└─ README.md           # Este arquivo
