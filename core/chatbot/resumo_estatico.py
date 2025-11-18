# core/resumo_estatico.py
import pandas as pd
from pathlib import Path
from core.chatbot.insights import gerar_insights

def gerar_resumo(input_path: str | None = None, output_dir: str | None = None, df: pd.DataFrame | None = None):
    """
    Gera resumos numéricos, interpretativos e insights da base e-commerce (spec_sales).
    - input_path: CSV original (opcional)
    - output_dir: pasta onde salvar os arquivos
    - df: DataFrame já carregado (opcional)
    """
    if df is None:
        if input_path is None:
            raise ValueError("É necessário fornecer df ou input_path")
        df = pd.read_csv(input_path)

    if output_dir is None:
        output_dir = Path("data")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------
    # Resumo numérico
    # ------------------------------
    num_cols = df.select_dtypes(include="number").columns
    resumo_num = df[num_cols].describe().round(2) if len(num_cols) > 0 else pd.DataFrame()
    resumo_num_path = output_dir / "resumo_ecommerce.csv"
    resumo_num.to_csv(resumo_num_path)
    
    # ------------------------------
    # Resumo interpretativo
    # ------------------------------
    interpretacao = []
    if 'TotalPrice' in df.columns:
        interpretacao.append(f"💰 Receita total: {df['TotalPrice'].sum():.2f}")
        interpretacao.append(f"📊 Média de TotalPrice por cliente: {df['TotalPrice'].mean():.2f}")
    if 'Quantity' in df.columns:
        interpretacao.append(f"🛒 Total de produtos vendidos: {df['Quantity'].sum():.0f}")
        interpretacao.append(f"📈 Média de produtos por pedido: {df['Quantity'].mean():.2f}")
    if 'UnitPrice' in df.columns:
        interpretacao.append(f"💲 Produto mais caro: {df['UnitPrice'].max():.2f}")
        interpretacao.append(f"💲 Produto mais barato: {df['UnitPrice'].min():.2f}")
    
    interpretativo_path = output_dir / "resumo_ecommerce_textual.csv"
    pd.DataFrame({"Resumo interpretativo": interpretacao}).to_csv(interpretativo_path, index=False)

    # ------------------------------
    # Insights executivos
    # ------------------------------
    try:
        insights_text = gerar_insights(df)
        insights_lines = insights_text.split("\n\n")
    except Exception:
        insights_lines = ["Não foi possível gerar insights automaticamente."]
    
    insights_path = output_dir / "resumo_ecommerce_insights.csv"
    pd.DataFrame({"Insights executivos": insights_lines}).to_csv(insights_path, index=False)

    return {
        "resumo_num": resumo_num_path,
        "interpretativo": interpretativo_path,
        "insights": insights_path
    }
