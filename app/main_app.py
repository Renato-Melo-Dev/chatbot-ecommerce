import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import unicodedata
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Funções de utilidade
# -------------------------------

def normalize_text(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Palavras-chave e sinônimos
keywords = {
    "mais caro": ["caro", "valor maximo", "preco alto", "mais caro"],
    "mais barato": ["barato", "valor minimo", "preco baixo", "mais barato"],
    "receita": ["vendas", "faturamento", "lucro", "receita"],
    "promoção": ["desconto", "oferta", "sale", "promocao"],
    "quantidade": ["quantidade", "mais vendido", "top venda"],
}

def match_keyword(prompt):
    prompt_norm = normalize_text(prompt)
    for key, syns in keywords.items():
        for syn in syns:
            if syn in prompt_norm:
                return key
    return None

# Funções de respostas
def respond(df, prompt):
    key = match_keyword(prompt)
    if df is None or df.empty:
        return "Ainda não tenho dados para responder. Envie o CSV primeiro."
    
    if key == "mais caro":
        prod = df.groupby("Description")["UnitPrice"].max().idxmax()
        val = df.groupby("Description")["UnitPrice"].max().max()
        return f"O produto mais caro é '{prod}' com R${val:,.2f}."
    
    elif key == "mais barato":
        prod = df.groupby("Description")["UnitPrice"].min().idxmin()
        val = df.groupby("Description")["UnitPrice"].min().min()
        return f"O produto mais barato é '{prod}' com R${val:,.2f}."
    
    elif key == "receita":
        prod = df.groupby("Description")["TotalPrice"].sum().idxmax()
        val = df.groupby("Description")["TotalPrice"].sum().max()
        return f"O produto que gera mais receita é '{prod}' com R${val:,.2f}."
    
    elif key == "promoção":
        if "Promotion" in df.columns:
            prods = df[df["Promotion"]==1]["Description"].unique()
            return f"Produtos em promoção: {', '.join(prods[:10])}..."
        else:
            return "Não há informações de promoção no CSV."
    
    elif key == "quantidade":
        prod = df.groupby("Description")["Quantity"].sum().idxmax()
        val = df.groupby("Description")["Quantity"].sum().max()
        return f"O produto mais vendido é '{prod}' com {val} unidades."

    else:
        return "Não entendi sua pergunta. Pode reformular usando termos como 'mais caro', 'receita', 'promoção' ou 'quantidade'."

# -------------------------------
# Conexão SQLite
# -------------------------------

DB_PATH = "ecommerce_db.sqlite"

def create_sqlite_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

def execute_sql_file(conn, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()

def insert_df_to_table_sqlite(df, table_name, conn):
    df = df.drop_duplicates()
    df = df.where(pd.notnull(df), None)
    if not df.empty:
        df.to_sql(table_name, conn, if_exists="append", index=False)

# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Chatbot Vendas Inteligente", layout="wide")
st.title("🛒 Chatbot Vendas Inteligente")

# Histórico
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Olá! Envie um CSV de vendas e treine o modelo. 🙂"}
    ]

with st.sidebar:
    test_size = st.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)
    uploaded = st.file_uploader("Envie o CSV de vendas", type=["csv"])

tab_train, tab_chat = st.tabs(["📊 Treino & Métricas", "💬 Chat"])

# -------------------------------
# Treino
# -------------------------------

with tab_train:
    if uploaded:
        df = pd.read_csv(uploaded)
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        st.write("Prévia dos dados", df.head())

        # Conexão SQLite
        conn = create_sqlite_connection()

        # Executar arquivos .sql se existirem
        for sql_file in ["core/data/sor_sales.sql", "core/data/sot_sales.sql"]:
            execute_sql_file(conn, sql_file)

        # Popular SOR
        insert_df_to_table_sqlite(df, "sor_sales", conn)

        # Criar SOT simples
        df_sot = df.copy()
        df_sot["TotalPrice_log"] = np.log1p(df_sot["TotalPrice"].replace([np.inf, -np.inf], 0))
        df_sot["Quantity_norm"] = (df_sot["Quantity"] - df_sot["Quantity"].mean()) / df_sot["Quantity"].std()
        df_sot["UnitPrice_norm"] = (df_sot["UnitPrice"] - df_sot["UnitPrice"].mean()) / df_sot["UnitPrice"].std()
        if "InvoiceDate" in df_sot.columns:
            df_sot["InvoiceDate"] = pd.to_datetime(df_sot["InvoiceDate"], errors='coerce')
            df_sot["DayOfWeek"] = df_sot["InvoiceDate"].dt.dayofweek
            df_sot["HourOfDay"] = df_sot["InvoiceDate"].dt.hour
        insert_df_to_table_sqlite(df_sot, "sot_sales", conn)

        # Treino modelo simples
        X_cols = ["Quantity_norm","UnitPrice_norm","DayOfWeek","HourOfDay"] if "DayOfWeek" in df_sot.columns else ["Quantity_norm","UnitPrice_norm"]
        X = df_sot[X_cols]
        y = df_sot["TotalPrice_log"].replace([np.inf, -np.inf], 0).fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.subheader("📈 Métricas")
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

        # Salvar modelo e dados
        st.session_state.last_model = model
        st.session_state.last_df = df

    else:
        st.info("⬆️ Envie um CSV de vendas na barra lateral para começar.")

# -------------------------------
# Chat
# -------------------------------
with tab_chat:
    st.caption("Converse com o assistente sobre os dados de vendas.")

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Faça sua pergunta (ex.: Qual produto gera mais receita?)")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        ans = respond(st.session_state.get("last_df"), prompt)
        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)
