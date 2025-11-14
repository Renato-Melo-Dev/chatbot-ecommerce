# core/chatbot/insights.py
import pandas as pd

def gerar_insights(df: pd.DataFrame, top_n_products=5, top_n_clients=5) -> str:
    """
    Gera insights de e-commerce a partir do DataFrame spec_sales.
    Retorna uma string com insights interpretáveis.
    """
    if df.empty:
        return "O DataFrame está vazio. Não há dados para gerar insights."

    insights = []

    # Produto mais caro
    if "UnitPrice" in df.columns and "Description" in df.columns:
        max_price = df["UnitPrice"].max()
        produto_caro = df[df["UnitPrice"] == max_price]["Description"].values[0]
        insights.append(f"🔹 Produto mais caro: {produto_caro} custa {max_price:.2f}.")

    # Produto mais vendido (Quantity)
    if "Quantity" in df.columns and "Description" in df.columns:
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(top_n_products)
        insights.append(f"🔹 Top {top_n_products} produtos mais vendidos:\n{top_products.to_string()}")

    # Clientes que mais gastaram (TotalPrice)
    if "TotalPrice" in df.columns and "CustomerID" in df.columns:
        top_clients = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(top_n_clients)
        insights.append(f"🔹 Top {top_n_clients} clientes que mais gastaram:\n{top_clients.to_string()}")

    # País com maior faturamento
    if "Country" in df.columns and "TotalPrice" in df.columns:
        top_country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(1)
        country_name = top_country.index[0]
        country_total = top_country.iloc[0]
        insights.append(f"🔹 País com maior faturamento: {country_name}, totalizando {country_total:.2f}.")

    # Correlação Quantity x TotalPrice
    if "Quantity" in df.columns and "TotalPrice" in df.columns:
        corr = df["Quantity"].corr(df["TotalPrice"])
        insights.append(f"🔹 Correlação entre Quantity e TotalPrice: {corr:.2f} (quanto maior a quantidade, maior a receita).")

    return "\n\n".join(insights)
