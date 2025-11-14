from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def train_regressor(X, y, preprocessor, test_size=0.2, random_state=42):
    """
    Treina um modelo Ridge Regression usando Pipeline com pré-processamento.
    Retorna:
        - pipeline treinada
        - X_test cru
        - y_test
    """
    # divide treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # define modelo Ridge
    model_pipe = Pipeline([
        ("pre", preprocessor),
        ("reg", Ridge(alpha=1.0, random_state=random_state))
    ])

    # treina o pipeline
    model_pipe.fit(X_train, y_train)

    return model_pipe, X_test, y_test
