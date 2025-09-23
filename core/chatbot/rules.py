# core/chatbot/rules.py
import pandas as pd
import numpy as np

def answer_from_metrics(question, metrics_df_or_dict=None, importances_df=None, model_pipe=None):
    """
    Responde perguntas sobre métricas, importâncias e previsões.
    """
    question = question.lower().strip()

    # 1️⃣ Pergunta sobre RMSE ou métricas
    if "rmse" in question:
        if metrics_df_or_dict is not None:
            rmse = metrics_df_or_dict.get("rmse") if isinstance(metrics_df_or_dict, dict) else None
            if rmse is not None:
                return f"✅ O RMSE do modelo é {rmse:.4f}."
            else:
                return "⚠️ Não encontrei RMSE nas métricas."
        return "⚠️ Métricas do modelo não estão disponíveis."

    # 2️⃣ Pergunta sobre top features
    if "features mais importantes" in question or "top features" in question:
        if importances_df is not None and not importances_df.empty:
            # Tenta encontrar a coluna de coeficientes
            coef_col = None
            for col in importances_df.columns:
                if "coef" in col.lower() or "importance" in col.lower():
                    coef_col = col
                    break

            if coef_col is None:
                return "⚠️ Coluna de coeficientes não encontrada no DataFrame de importâncias."

            # Calcular porcentagem relativa se não existir
            if "Relativa (%)" not in importances_df.columns:
                importances_df["Relativa (%)"] = (importances_df[coef_col].abs() / importances_df[coef_col].abs().sum()) * 100

            top_features = importances_df.copy()
            top_features["abs_coef"] = top_features[coef_col].abs()
            top_features = top_features.sort_values(by="abs_coef", ascending=False).head(5)

          # Mostrar Feature + %
            features_list = ", ".join([f"{f} ({p:.1f}%)" 
                                    for f, p in zip(top_features.iloc[:, 0], top_features["Relativa (%)"])])
            return f"🔎 As 5 features mais importantes são: {features_list}."


    # 3️⃣ Pergunta sobre previsão de cliente/país
    if "previsão" in question or "quanto seria" in question:
        if model_pipe is not None:
            return ("Para gerar uma previsão, forneça CustomerID e Country "
                    "no formato: CustomerID=xxx, Country=YYY")
        return "⚠️ Modelo não está carregado para fazer previsões."

    # Caso a pergunta não seja reconhecida
    return ("❓ Desculpe, não entendi a pergunta. "
            "Você pode perguntar sobre RMSE, features ou previsões de TotalPrice.")