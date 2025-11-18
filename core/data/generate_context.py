import pandas as pd
from pathlib import Path
from typing import Union

from core.config import DATA_DIR, CONTEXT_CACHE_DIR, logger


def generate_context(summary_type: str = "full") -> Union[str, Path]:
    """Gera um resumo textual da base `eCommerce.csv` para uso como contexto RAG.

    Args:
        summary_type: 'full', 'describe' ou 'sample' para controlar o que é gerado.

    Returns:
        Caminho para o arquivo de contexto gerado.
    """
    csv_path = Path(DATA_DIR) / "eCommerce.csv"
    context_path = Path(CONTEXT_CACHE_DIR) / f"context_{summary_type}.txt"

    if not csv_path.exists():
        logger.error(f"Arquivo CSV não encontrado: {csv_path}")
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    # Corrige colunas comuns
    df.columns = [col.strip() for col in df.columns]

    # Calcula TotalPrice
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    text_blocks = []

    # 1) Resumo estatístico
    if summary_type in ["full", "describe"]:
        text_blocks.append("### RESUMO ESTATÍSTICO\n")
        text_blocks.append(df.describe(include="all").to_string())

    # 2) Amostra da tabela
    if summary_type in ["full", "sample"]:
        text_blocks.append("\n\n### PRIMEIRAS LINHAS\n")
        text_blocks.append(df.head(20).to_string())

    # 3) Top países por vendas
    if "Country" in df.columns and "TotalPrice" in df.columns:
        country_sales = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
        text_blocks.append("\n\n### FATURAMENTO POR PAÍS\n")
        text_blocks.append(country_sales.to_string())

    # 4) Top produtos
    if "Description" in df.columns and "Quantity" in df.columns:
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(20)
        text_blocks.append("\n\n### TOP 20 PRODUTOS MAIS VENDIDOS\n")
        text_blocks.append(top_products.to_string())

    # 5) Top clientes
    if "CustomerID" in df.columns and "TotalPrice" in df.columns:
        top_customers = df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(20)
        text_blocks.append("\n\n### TOP 20 CLIENTES QUE MAIS GASTARAM\n")
        text_blocks.append(top_customers.to_string())

    # 6) Insights automáticos
    insights = []
    if "UnitPrice" in df.columns:
        max_price = df["UnitPrice"].max()
        prod_max = df[df["UnitPrice"] == max_price]["Description"].head(1).values
        insights.append(f"- Produto mais caro custa {max_price:.2f} ({prod_max[0] if len(prod_max) else 'N/A'}).")

    if "Quantity" in df.columns:
        avg_q = df["Quantity"].mean()
        insights.append(f"- Quantidade média por item: {avg_q:.2f}.")

    if "Country" in df.columns:
        best_country = country_sales.index[0]
        insights.append(f"- País com maior faturamento: {best_country}.")

    text_blocks.append("\n\n### INSIGHTS\n" + "\n".join(insights))

    # Salvando o contexto
    final_context = "\n".join(text_blocks)
    CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(final_context)

    logger.info(f"Contexto gerado: {context_path}")
    return context_path
