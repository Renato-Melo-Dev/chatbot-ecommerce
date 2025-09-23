# core/chatbot/rules.py
import pandas as pd

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
            coef_col = None
            for col in importances_df.columns:
                if "coef" in col.lower() or "importance" in col.lower():
                    coef_col = col
                    break

            if coef_col is None:
                return "⚠️ Coluna de coeficientes não encontrada no DataFrame de importâncias."

            # Calcular % relativo e ordenar
            df_imp = importances_df.copy()
            total_abs = df_imp[coef_col].abs().sum()
            df_imp["Importância (%)"] = (df_imp[coef_col].abs() / total_abs * 100).round(1)

            # Usar Description se existir, senão usa a primeira coluna
            if "Description" in df_imp.columns:
                feature_name_col = "Description"
            else:
                feature_name_col = df_imp.columns[0]

            top_features = df_imp.sort_values(by="Importância (%)", ascending=False).head(5)
            features_list = ", ".join(top_features[feature_name_col].astype(str).tolist())
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
            "Você pode perguntar sobre RMSE, features ou previsões de TotalPrice.")
