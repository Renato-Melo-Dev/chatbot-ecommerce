# core/chatbot/rules.py
import pandas as pd
import numpy as np

# Mapeamento de países PT → EN (para o chat interpretar)
COUNTRY_MAP = {
    "alemanha": "Germany",
    "reino unido": "United Kingdom",
    "islandia": "Iceland",
    "brasil": "Brazil",
    "franca": "France",
    "italia": "Italy",
    # adicione conforme necessário
}

def generate_insights(df: pd.DataFrame, top_n=5, metric="TotalQuantity"):
    """
    Gera insights automáticos sobre produtos agregados.
    df deve conter colunas: Country, StockCode, Description, TotalQuantity, AvgUnitPrice, TotalRevenue, NumInvoices, NumCustomers
    metric: "TotalQuantity" ou "AvgUnitPrice" ou "TotalRevenue"
    """
    df_sorted = df.sort_values(metric, ascending=False).head(top_n)
    insights = []
    for _, row in df_sorted.iterrows():
        insights.append(
            f"Produto: {row['Description']} (Código: {row['StockCode']})\n"
            f"País: {row['Country']}\n"
            f"Quantidade vendida: {row['TotalQuantity']}, Receita total: {row['TotalRevenue']:.2f}, "
            f"Preço médio: {row['AvgUnitPrice']:.2f}, Clientes distintos: {row['NumCustomers']}, Invoices: {row['NumInvoices']}\n"
        )

    # Insights gerais
    total_qty = df_sorted['TotalQuantity'].sum()
    total_rev = df_sorted['TotalRevenue'].sum()
    avg_price = df_sorted['AvgUnitPrice'].mean()
    insights.append(
        f"💡 Insights: Top {top_n} produtos representam {total_qty} unidades vendidas no total "
        f"com receita de {total_rev:.2f} e preço médio unitário de {avg_price:.2f}."
    )
    return "\n".join(insights)

def answer_from_metrics(question, metrics_df_or_dict=None, importances_df=None, model_pipe=None, df_spec=None):
    question_lower = question.lower().strip()

    # Perguntas sobre métricas do modelo
    if "rmse" in question_lower or "mae" in question_lower or "r2" in question_lower:
        if metrics_df_or_dict:
            metrics_text = "\n".join([f"{k}: {v}" for k,v in metrics_df_or_dict.items()])
            return f"📊 Métricas do modelo:\n{metrics_text}"
        return "⚠️ Métricas do modelo não estão disponíveis."

    # Perguntas sobre top produtos
    if "top produtos" in question_lower or "produtos mais vendidos" in question_lower or "produtos mais caros" in question_lower:
        if df_spec is None:
            return "⚠️ Dados não disponíveis para gerar ranking de produtos."

        # Detectar país
        country = None
        for key in COUNTRY_MAP.keys():
            if key in question_lower:
                country = COUNTRY_MAP[key]
                break
        df_filtered = df_spec[df_spec["Country"] == country] if country else df_spec

        if df_filtered.empty:
            return f"⚠️ Não há dados para o país {country}." if country else "⚠️ Não há dados disponíveis."

        # Detectar número de produtos solicitados
        import re
        top_n = 5
        match = re.search(r'\btop\s+(\d+)', question_lower)
        if match:
            top_n = int(match.group(1))

        # Definir métrica
        if "mais caro" in question_lower:
            metric = "AvgUnitPrice"
        elif "mais vendido" in question_lower:
            metric = "TotalQuantity"
        elif "maior receita" in question_lower or "top revenue" in question_lower:
            metric = "TotalRevenue"
        else:
            metric = "TotalQuantity"

        return generate_insights(df_filtered, top_n=top_n, metric=metric)

    # Previsão específica de cliente
    if "previsão cliente" in question_lower:
        import re
        if model_pipe is None or df_spec is None:
            return "⚠️ Modelo ou dados não disponíveis para gerar previsão."
        m = re.search(r'customerid=(\S+).*stockcode=(\S+)', question_lower)
        if not m:
            return "⚠️ Informe no formato: CustomerID=xxx, StockCode=yyy"
        cust_id, stock = m.group(1), m.group(2)
        df_client = df_spec[(df_spec["CustomerID"]==float(cust_id)) & (df_spec["StockCode"]==stock)]
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

    return ("❓ Desculpe, não entendi a pergunta. "
            "Você pode perguntar sobre métricas (RMSE, MAE, R2), "
            "top produtos mais vendidos, mais caros, ou previsões específicas de cliente.")
