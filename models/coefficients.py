# models/coefficients.py
import pandas as pd

def extract_linear_importances(model_pipe, feature_names, preprocessor):
    """
    Extrai os coeficientes do LinearRegression após OneHotEncoder
    e calcula a importância relativa em %.
    """
    # Separar transformadores
    ohe = preprocessor.named_transformers_['cat']['onehot']
    num_cols = preprocessor.transformers_[1][2]  # ['Quantity', 'UnitPrice']

    # Obter nomes das colunas codificadas
    ohe_features = list(ohe.get_feature_names_out(preprocessor.transformers_[0][2]))
    all_features = ohe_features + num_cols

    # Coeficientes
    coefs = model_pipe.named_steps['reg'].coef_

    # Criar DataFrame
    df_importances = pd.DataFrame({
        "Feature": all_features,
        "Coeficiente": coefs
    })

    # Calcular porcentagem relativa (valor absoluto)
    df_importances["Relativa (%)"] = (df_importances["Coeficiente"].abs() / df_importances["Coeficiente"].abs().sum()) * 100

    # Ordenar por valor absoluto
    df_importances = df_importances.sort_values(by="Coeficiente", key=abs, ascending=False)

    return df_importances