# core/pipeline.py
import pickle
import time
import pandas as pd
import numpy as np
from core.database import (
    create_database_and_tables,
    insert_csv_to_sor,
    run_etl_sor_to_sot,
    load_data
)
from core.preprocess import make_preprocess_pipeline
from models.train import train_regressor
from models.coefficients import extract_linear_importances

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
]

def fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas e preenche as faltantes."""
    df.columns = [c.strip() for c in df.columns]
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "CustomerID":
                df[col] = "Desconhecido"
            elif col in ["Quantity", "UnitPrice"]:
                df[col] = 0
            else:
                df[col] = "Desconhecido"
        else:
            if col == "CustomerID":
                df[col] = df[col].fillna("Desconhecido")
            elif col in ["Quantity", "UnitPrice"]:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Desconhecido")
    return df

# --- ETL para treino ---
def run_etl_sot_to_spec_train():
    from core.database import engine
    query = """
    SELECT CustomerID, Country, StockCode, Description,
           SUM(Quantity) AS Quantity,
           AVG(UnitPrice) AS UnitPrice,
           SUM(Quantity * UnitPrice) AS TotalPrice
    FROM SOT
    GROUP BY CustomerID, Country, StockCode, Description
    """
    df = pd.read_sql(query, engine)
    df.to_sql("spec_sales", engine, if_exists="replace", index=False)

# --- ETL para teste ---
def run_etl_for_test_data(df_test: pd.DataFrame):
    from core.database import engine

    df_test = fill_missing_columns(df_test)
    df_test.to_sql("SOR_TEST", engine, if_exists="replace", index=False)

    query = """
    SELECT CustomerID, Country, StockCode, Description,
           SUM(Quantity) AS Quantity,
           AVG(UnitPrice) AS UnitPrice,
           SUM(Quantity * UnitPrice) AS TotalPrice
    FROM SOR_TEST
    GROUP BY CustomerID, Country, StockCode, Description
    """
    df_sot_test = pd.read_sql(query, engine)
    df_sot_test.to_sql("spec_sales", engine, if_exists="replace", index=False)

# --- Função de avaliação custom ---
def evaluate_regressor_custom(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2}

# --- Pipeline de treino ---
def run_training_pipeline(df_train: pd.DataFrame, test_size: float, model_path: str):
    df_train = fill_missing_columns(df_train)

    create_database_and_tables()
    time.sleep(0.2)

    insert_csv_to_sor(df_train)
    time.sleep(0.2)

    run_etl_sor_to_sot()
    run_etl_sot_to_spec_train()
    time.sleep(0.2)

    df_spec = load_data("spec_sales")
    if df_spec.empty:
        raise ValueError("Tabela spec_sales está vazia.")

    target = "TotalPrice"
    if target not in df_spec.columns:
        df_spec[target] = df_spec["Quantity"] * df_spec["UnitPrice"]

    feature_cols = ["CustomerID", "Country", "StockCode", "Quantity", "UnitPrice"]
    X = df_spec[feature_cols]
    y = np.log1p(df_spec[target])  # log-transform para evitar negativos

    pre = make_preprocess_pipeline(X)
    model_pipe, X_test, y_test = train_regressor(X, y, pre, test_size=test_size)

    with open(model_path, "wb") as f:
        pickle.dump(model_pipe, f)

    # Avaliar usando valores originais
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(model_pipe.predict(X_test))
    metrics = evaluate_regressor_custom(y_test_orig, y_pred_orig)

    importances = extract_linear_importances(model_pipe, X.columns, pre)

    return metrics, importances

# --- Pipeline de predição ---
def run_prediction_pipeline(df_test: pd.DataFrame, model_path: str):
    df_test = fill_missing_columns(df_test)

    run_etl_for_test_data(df_test)
    df_spec_predict = load_data("spec_sales")
    if df_spec_predict.empty:
        raise ValueError("Tabela spec_sales está vazia.")

    feature_cols = ["CustomerID", "Country", "StockCode", "Quantity", "UnitPrice"]
    for col in feature_cols:
        if col not in df_spec_predict.columns:
            df_spec_predict[col] = 0 if col in ["Quantity", "UnitPrice", "CustomerID"] else "Desconhecido"

    X_predict = df_spec_predict[feature_cols]

    with open(model_path, "rb") as f:
        model_pipe = pickle.load(f)

    predictions_log = model_pipe.predict(X_predict)
    predictions = np.expm1(predictions_log)  # volta ao valor original
    predictions = np.maximum(predictions, 0)  # garante que não seja negativo

    df_pred = df_spec_predict[feature_cols + ["Description"]].copy()  # Mantém Description
    df_pred["Predito"] = predictions

    return df_pred
