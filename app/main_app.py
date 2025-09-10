import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------
# Banco de dados
# -----------------------
def create_database(db_name="ecommerce.db"):
    if os.path.exists(db_name):
        os.remove(db_name)  # dropa o banco se existir
    engine = create_engine(f"sqlite:///{db_name}")
    st.info(f"Banco de dados '{db_name}' carregado ✅")
    return engine

# -----------------------
# Executar scripts SQL
# -----------------------
def run_sql_scripts(engine, sql_folder="core/sql"):
    sql_path = Path(sql_folder)
    if not sql_path.exists():
        st.warning(f"Pasta SQL '{sql_folder}' não encontrada.")
        return
    sql_files = sql_path.glob("*.sql")
    with engine.begin() as conn:
        for sql_file in sql_files:
            with open(sql_file, "r", encoding="utf-8") as f:
                script = f.read()
                commands = [cmd.strip() for cmd in script.split(";") if cmd.strip()]
                for command in commands:
                    conn.execute(text(command))
    st.success("Scripts SQL executados ✅")

# -----------------------
# Carregar CSV e criar tabelas
# -----------------------
def load_data_to_db(engine, csv_file):
    df = pd.read_csv(csv_file)
    st.info(f"CSV carregado ✅")
    
    df.to_sql("SOR", engine, if_exists="replace", index=False)

    df_sot = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].dropna(subset=["CustomerID"])
    st.info(f"Registros SOT: {len(df_sot)} / {len(df)}")
    df_sot.to_sql("SOT", engine, if_exists="replace", index=False)

    df_spec = df_sot.copy()
    df_spec["TotalPrice"] = df_spec["Quantity"] * df_spec["UnitPrice"]
    df_spec["InvoiceDate"] = pd.to_datetime(df_spec["InvoiceDate"], errors="coerce")
    df_spec.to_sql("SPEC", engine, if_exists="replace", index=False)

    return df_spec

# -----------------------
# Treinar modelo
# -----------------------
def train_model(df_spec):
    X = df_spec[["Quantity","UnitPrice"]]
    y = df_spec["TotalPrice"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_pred)
    return model, y_test, y_pred, mae, mse, rmse, r2

# -----------------------
# Previsão com modelo salvo
# -----------------------
def predict_from_pickle(engine, pickle_path="modelo.pkl"):
    if not os.path.exists(pickle_path):
        st.error("Modelo não encontrado. Treine um modelo primeiro.")
        return pd.DataFrame()
    with open(pickle_path,"rb") as f:
        model = pickle.load(f)
    df_spec = pd.read_sql("SELECT Quantity, UnitPrice, TotalPrice, CustomerID, Country, InvoiceDate FROM SPEC", engine)
    if df_spec.empty:
        st.warning("Tabela SPEC vazia.")
        return pd.DataFrame()
    df_spec["InvoiceDate"] = pd.to_datetime(df_spec["InvoiceDate"], errors="coerce")
    X = df_spec[["Quantity","UnitPrice"]]
    df_spec["Predito"] = model.predict(X)
    return df_spec

# -----------------------
# Dashboard Streamlit
# -----------------------
def main():
    st.set_page_config(page_title="E-commerce Dashboard", layout="wide")
    st.title("📊 Dashboard E-commerce com Previsão")

    # -------------------------------
    # Upload do CSV
    # -------------------------------
    uploaded_file = st.file_uploader("📁 Faça upload do CSV de e-commerce", type=["csv"])
    if uploaded_file is None:
        st.warning("Aguardando o upload do CSV para continuar...")
        st.stop()  # interrompe a execução até que o CSV seja enviado

    # -------------------------------
    # Criar engine e carregar dados
    # -------------------------------
    engine = create_database()
    run_sql_scripts(engine)
    df_spec = load_data_to_db(engine, uploaded_file)

    if df_spec.empty:
        st.error("O CSV carregado não contém dados válidos.")
        st.stop()

    # -------------------------------
    # Botões para Treinar ou Carregar
    # -------------------------------
    st.subheader("O que deseja fazer?")
    col1, col2 = st.columns(2)
    train_clicked = col1.button("🚀 Treinar modelo")
    load_clicked = col2.button("📦 Carregar modelo")

    df_pred = pd.DataFrame()  # inicializa vazio

    # -------------------------------
    # Treinar modelo
    # -------------------------------
    if train_clicked:
        st.info("Treinando novo modelo...")
        model, y_test, y_pred, mae, mse, rmse, r2 = train_model(df_spec)
        with open("modelo.pkl","wb") as f:
            pickle.dump(model,f)
        st.success("Modelo treinado e salvo ✅")

        # Layout de métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("R²", f"{r2:.2f}")

        # Gráfico Real x Previsto
        st.plotly_chart(px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "TotalPrice Real", "y": "TotalPrice Previsto"},
            title="Real x Previsto (Treino)"
        ), use_container_width=True)

        # Predição completa
        df_pred = predict_from_pickle(engine)

    # -------------------------------
    # Carregar modelo
    # -------------------------------
    elif load_clicked:
        st.info("Carregando modelo salvo...")
        df_pred = predict_from_pickle(engine)
        if not df_pred.empty:
            st.success("Predições realizadas com modelo existente ✅")
            st.metric("Total Predito", f"${df_pred['Predito'].sum():,.2f}")

    # -------------------------------
    # Dashboard principal
    # -------------------------------
    if not df_pred.empty:
        st.subheader("📊 Indicadores e Filtros")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vendas", f"${df_pred['TotalPrice'].sum():,.2f}")
            st.metric("Média TotalPrice", f"${df_pred['TotalPrice'].mean():,.2f}")
        with col2:
            st.metric("Maior Venda", f"${df_pred['TotalPrice'].max():,.2f}")
            st.metric("Quantidade de Clientes", f"{df_pred['CustomerID'].nunique()}")
        with col3:
            st.metric("Vendas Previstas Totais", f"${df_pred['Predito'].sum():,.2f}")

        # Filtros
        st.subheader("Filtros")
        countries = st.multiselect("Selecione países", df_pred["Country"].unique(), default=df_pred["Country"].unique())
        customers = st.multiselect("Selecione clientes", df_pred["CustomerID"].unique(), default=df_pred["CustomerID"].unique())
        dates = st.date_input(
            "Selecione intervalo de datas",
            [df_pred["InvoiceDate"].min().date(), df_pred["InvoiceDate"].max().date()]
        )
        if isinstance(dates, (tuple, list)) and len(dates) == 2:
            start_date, end_date = dates
        else:
            start_date = end_date = dates if isinstance(dates, pd.Timestamp) else pd.to_datetime(dates[0])
        mask = (df_pred["Country"].isin(countries)) & \
               (df_pred["CustomerID"].isin(customers)) & \
               (df_pred["InvoiceDate"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))
        df_filtered = df_pred[mask]

        # Tabelas e gráficos
        st.subheader("📋 Tabela de vendas filtrada")
        st.dataframe(df_filtered.head(20))

        st.subheader("🌍 TotalPrice por País")
        fig1 = px.bar(df_filtered.groupby("Country")["TotalPrice"].sum().reset_index(), x="Country", y="TotalPrice")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🏆 Top Clientes por TotalPrice")
        top_customers = df_filtered.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(top_customers, x="CustomerID", y="TotalPrice")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔹 Real x Previsto")
        fig3 = px.scatter(df_filtered, x="TotalPrice", y="Predito", color="Country", title="Real x Previsto")
        st.plotly_chart(fig3, use_container_width=True)

if __name__=="__main__":
    main()
