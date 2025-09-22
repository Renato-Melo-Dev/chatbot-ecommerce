# models/coefficients.py
import pandas as pd

def extract_linear_importances(model_pipe, feature_names, preprocessor):
    """
    Extrai os coeficientes do LinearRegression após OneHotEncoder.
    Retorna um DataFrame com colunas 'feature' e 'importance', compatível com rules.py.
    """
    # Transformador de categóricas
    ohe = preprocessor.named_transformers_['cat']['onehot']
    # Colunas numéricas
    num_cols = preprocessor.transformers_[1][2]  # ['Quantity', 'UnitPrice']

    # Obter nomes das colunas codificadas
    ohe_features = list(ohe.get_feature_names_out(preprocessor.transformers_[0][2]))
    all_features = ohe_features + num_cols

    # Coeficientes do modelo
    coefs = model_pipe.named_steps['reg'].coef_

    # Criar DataFrame compatível com rules.py
    df_importances = pd.DataFrame({
        "feature": all_features,
        "importance": coefs
    }).sort_values(by="importance", key=abs, ascending=False)

    return df_importances
