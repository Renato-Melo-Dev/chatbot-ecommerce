# core/chatbot/rules.py
def answer_from_metrics(question: str, task: str = "regressão", metrics_df_or_dict=None, importances_df=None, df=None, model_pipe=None):
    q = (question or "").lower()

    # 1️⃣ Pergunta sobre métricas
    if "rmse" in q or "r²" in q or "r2" in q or "mae" in q or "métric" in q:
        if metrics_df_or_dict:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics_df_or_dict.items()])
            return f"Métricas da tarefa {task}: {metrics_str}"
        return "⚠️ Métricas não estão disponíveis."

    # 2️⃣ Pergunta sobre features
    if "feature" in q or "variável" in q:
        features = ["CustomerID", "Country", "StockCode", "Quantity", "UnitPrice", "Month", "Year"]
        return f"🔧 As features usadas pelo modelo são: {', '.join(features)}. Importâncias detalhadas não estão disponíveis."

    # 3️⃣ Pergunta sobre pipeline
    if "pipeline" in q or "treinado" in q or "como foi treinado" in q:
        return "O pipeline faz imputação de valores faltantes, criação de Month/Year, one-hot encoding de variáveis categóricas e treina RandomForestRegressor."

    # 4️⃣ Pergunta sobre previsão
    if "previsão" in q or "quanto seria" in q:
        if model_pipe is not None:
            return ("Para gerar uma previsão, forneça CustomerID, Country, StockCode, Quantity e UnitPrice "
                    "no formato: CustomerID=xxx, Country=YYY, StockCode=ZZZ, Quantity=N, UnitPrice=P")
        return "⚠️ Modelo não carregado para fazer previsões."

    # 5️⃣ Pergunta sobre número de clientes
    if "cliente" in q or "quantos clientes" in q:
        if df is not None and "CustomerID" in df.columns:
            n_clients = df["CustomerID"].nunique()
            return f"👥 Existem {n_clients} clientes únicos no dataset."
        return "⚠️ Dataset não disponível para contar clientes."

    # Pergunta não reconhecida
    return ("❓ Desculpe, não entendi a pergunta. "
            "Você pode perguntar sobre métricas, features, pipeline, previsões ou número de clientes.")
