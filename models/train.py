from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def train_regressor(X, y, preprocessor, test_size=0.2):
    """
    Treina regressão linear usando Pipeline com pré-processamento.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_pipe = Pipeline([
        ("pre", preprocessor),
        ("reg", LinearRegression())
    ])

    model_pipe.fit(X_train, y_train)

    # X_test permanece cru, o Pipeline transforma internamente
    return model_pipe, X_test, y_test
