from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_regressor(model_pipe, X_test, y_test):
    """
    Avalia regressão linear com métricas principais:
    RMSE, MAE e R².
    """
    # Previsão
    y_pred = model_pipe.predict(X_test)

    # Cálculo das métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }

    return metrics
