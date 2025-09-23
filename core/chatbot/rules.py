# core/chatbot/rules.py
import pandas as pd

def answer_from_metrics(question, metrics_df_or_dict=None, importances_df=None, model_pipe=None):
    """
    Responde perguntas sobre métricas, importâncias e previsões.
    """
    question = question.lower().strip()

    # 1️⃣ Pergunta sobre RMSE ou métricas
    if "rmse" in question or "mae" in question or "r2" in question:
        if metrics_df_or_dict is not None:
            rmse = metrics_df_or_dict.get("rmse")
            mae = metrics_df_or_dict.get("mae")
            r2 = metrics_df_or_dict.get("r2")
            return (f"📊 Métricas do modelo:\n"
                    f"- RMSE: {rmse:.4f}\n"
                    f"- MAE: {mae:.4f}\n"
                    f"- R²: {r2:.4f}")
        return "⚠️ Métricas do modelo não estão disponíveis."

    # 2️⃣ Pergunta sobre top features
    if "features mais importantes" in question or "top features" in question:
        if importances_df is not None and not importances_df.empty:
            top_features = importances_df.head(5)
            features_list = ", ".join(top_features["Feature"].astype(str).tolist())
            return f"🔎 As 5 features mais importantes são: {features_list}."
        return "⚠️ Importâncias das features não estão disponíveis."

    # 3️⃣ Pergunta sobre previsão
    if "previsão" in question or "quanto seria" in question:
        if model_pipe is not None:
            return ("Para gerar uma previsão, forneça CustomerID e Country "
                    "no formato: CustomerID=xxx, Country=YYY")
        return "⚠️ Modelo não está carregado para fazer previsões."

    # Pergunta não reconhecida
    return ("❓ Desculpe, não entendi a pergunta. "
            "Você pode perguntar sobre RMSE, top features ou previsões de TotalPrice.")
