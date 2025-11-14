# core/chatbot/rules.py
import numpy as np

def answer_from_metrics(question, metrics_df_or_dict=None, importances_df=None, model_pipe=None, df_spec=None):
    question = question.lower().strip()

    # RMSE ou outras métricas
    if "rmse" in question:
        if metrics_df_or_dict:
            rmse = metrics_df_or_dict.get("rmse")
            if rmse is not None:
                return f"✅ O RMSE do modelo é {rmse:.4f}."
        return "⚠️ Métricas do modelo não estão disponíveis."

    # Previsão específica de cliente
    if "previsão cliente" in question:
        if model_pipe is None or df_spec is None:
            return "⚠️ Modelo ou dados não disponíveis para gerar previsão."

        import re
        m = re.search(r'customerid=(\S+).*stockcode=(\S+)', question)
        if not m:
            return "⚠️ Informe no formato: CustomerID=xxx, StockCode=yyy"

        cust_id, stock = m.group(1), m.group(2)
        df_client = df_spec[(df_spec["CustomerID"]==cust_id) & (df_spec["StockCode"]==stock)]
        if df_client.empty:
            return f"⚠️ Cliente {cust_id} ou produto {stock} não encontrado."

        feature_cols = ["CustomerID", "Country", "StockCode", "Quantity", "UnitPrice", "Month", "Year"]
        X_example = df_client[feature_cols]
        y_pred_log = model_pipe.predict(X_example)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)

        desc = df_client["Description"].values[0]
        qty = df_client["Quantity"].values[0]
        return (f"Cliente {cust_id}, Produto: {desc}, Quantidade: {qty}\n"
                f"💰 TotalPrice previsto: {y_pred[0]:.2f}")

    # Perguntas não reconhecidas
    return ("❓ Desculpe, não entendi a pergunta. "
            "Você pode perguntar sobre RMSE, top features ou previsões de TotalPrice.\n"
            "Para previsão específica: CustomerID=xxx, StockCode=yyy")
