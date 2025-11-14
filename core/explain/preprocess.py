# core/preprocess.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def make_preprocess_pipeline(df):
    """
    Cria pipeline de pré-processamento.
    Recebe DataFrame com colunas categóricas e numéricas:
    - Categóricas: CustomerID, Country, StockCode → OneHotEncoder
    - Numéricas: Quantity, UnitPrice → passam sem transformação
    """
    # Colunas categóricas
    categorical_cols = ["CustomerID", "Country", "StockCode"]
    # Colunas numéricas
    numeric_cols = ["Quantity", "UnitPrice"]

    # Pipeline para categóricas
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer combinando pipelines
    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, categorical_cols),
        ("num", "passthrough", numeric_cols)
    ])

    return preprocessor
