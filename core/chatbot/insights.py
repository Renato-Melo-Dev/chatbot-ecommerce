# core/chatbot/insights.py
import pandas as pd
import re

def _match_country(user_text: str, df_countries) -> str | None:
    """Detecta país mencionado no texto."""
    match = re.search(r"(na|no|em|para|da|do)\s+([a-zA-Z ]+)", user_text)
    if match:
        country_raw = match.group(2).strip().lower()
    else:
        country_raw = None
        for c in df_countries:
            if c.lower() in user_text.lower():
                return c
    
    if not country_raw:
        return None

    for c in df_countries:
        if country_raw in c.lower():
            return c

    return None


def gerar_insights(
    df: pd.DataFrame,
    pergunta: str | None = None,
    country: str | None = None,
    top_n_products=5,
    top_n_clients=5,
) -> str:
    """
    Gera insights interpretáveis sobre e-commerce.
    Agora aceita explicitamente:
      - country="Spain"
      - top_n_products=5
      - top_n_clients=5
    """

    if df is None or df.empty:
        return "O DataFrame está vazio ou não foi carregado."

    insights = []

    # === 1) Se o país foi passado explicitamente, use ele ===
    detected_country = None

    if country:
        detected_country = country

    # === 2) Se NÃO foi passado, tente detectar pela pergunta ===
    elif pergunta and "country" in df.columns:
        df_countries = df["Country"].dropna().unique()
        detected_country = _match_country(pergunta, df_countries)

    # === 3) Se um país foi detectado/pedido, filtrar ===
    if detected_country:
        df = df[df["Country"] == detected_country]
        if df.empty:
            return f"Não há dados disponíveis para o país '{detected_country}'."
        insights.append(f"📍 Insights filtrados para o país: **{detected_country}**")

    # -------------------------------------------------------------------

    # 1) Produto mais caro
    if {"UnitPrice", "Description"}.issubset(df.columns):
        max_price = df["UnitPrice"].max()
        produto_caro = df.loc[df["UnitPrice"] == max_price, "Description"].iloc[0]
        insights.append(f"🔹 **Produto mais caro:** {produto_caro} — {max_price:.2f}")

    # 2) Top produtos
    if {"Quantity", "Description"}.issubset(df.columns):
        top_products = (
            df.groupby("Description")["Quantity"]
              .sum()
              .sort_values(ascending=False)
              .head(top_n_products)
        )
        insights.append(f"🔹 **Top {top_n_products} produtos mais vendidos:**\n{top_products.to_string()}")

    # 3) Top clientes
    if {"TotalPrice", "CustomerID"}.issubset(df.columns):
        top_clients = (
            df.groupby("CustomerID")["TotalPrice"]
              .sum()
              .sort_values(ascending=False)
              .head(top_n_clients)
        )
        insights.append(f"🔹 **Top {top_n_clients} clientes que mais gastaram:**\n{top_clients.to_string()}")

    # 4) País com maior faturamento (somente se NÃO filtrou)
    if not detected_country and {"Country", "TotalPrice"}.issubset(df.columns):
        top_country = (
            df.groupby("Country")["TotalPrice"]
              .sum()
              .sort_values(ascending=False)
              .head(1)
        )
        c_name = top_country.index[0]
        c_total = top_country.iloc[0]
        insights.append(f"🔹 **País com maior faturamento:** {c_name} — total de {c_total:.2f}")

    # 5) Correlação
    if {"Quantity", "TotalPrice"}.issubset(df.columns):
        corr = df["Quantity"].corr(df["TotalPrice"])
        insights.append(
            f"🔹 **Correlação Quantity × TotalPrice:** {corr:.2f}\n"
            "(quanto maior a quantidade vendida, maior tende a ser a receita)."
        )

    if not insights:
        return "Não foi possível gerar insights para essa pergunta."

    return "\n\n".join(insights)
