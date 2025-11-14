# core/data/readcsv.py
import pandas as pd
import streamlit as st

def read_csv(uploaded_file):
    """
    Lê um arquivo CSV enviado pelo usuário via Streamlit ou caminho local.
    Aplica ajustes básicos de encoding, separador e tratamento de colunas.
    """
    if uploaded_file is None:
        st.warning("📁 Nenhum arquivo enviado.")
        return None

    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',')
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        return None

    # remover espaços nos nomes das colunas
    df.columns = [c.strip() for c in df.columns]

    # opcional: pode colocar outras validações aqui (colunas obrigatórias, tipos, etc.)
    REQUIRED_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
                        "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        st.warning(f"Colunas ausentes no CSV: {missing_cols}")

    return df
