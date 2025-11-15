from core.chatbot.insights import gerar_insights

# Lista simples para detectar países automaticamente
COUNTRIES = [
    "EIRE", "Germany", "France", "Portugal", "Spain",
    "United Kingdom", "Belgium", "Netherlands"
]

def detect_country(question: str):
    q = question.lower()
    for c in COUNTRIES:
        if c.lower() in q:
            return c
    return None


def answer_insights(question: str, df):
    """Responde perguntas sobre vendas, produtos, faturamento etc."""
    q = question.lower()

    # Gatilhos que identificam perguntas sobre dados do ecommerce
    gatilhos = [
        "venda", "produto", "faturamento", "país", "ecommerce",
        "compras", "clientes", "receita", "quantidade", "dados"
    ]

    if not any(g in q for g in gatilhos):
        return None

    country = detect_country(question)

    return gerar_insights(df, country=country)
