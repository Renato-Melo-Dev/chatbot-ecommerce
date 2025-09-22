import pandas as pd

def read_csv_smart(file, required_columns=None):
    """
    Lê um CSV de forma inteligente:
    - Detecta separador automaticamente
    - Detecta encoding
    - Preenche colunas faltantes com zeros ou valores padrão
    """
    try:
        # Tenta UTF-8, senão ISO-8859-1
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        
        # Detecta separador se necessário
        if df.shape[1] == 1 and ";" in df.columns[0]:
            df = pd.read_csv(file, sep=";")
        
        # Preenche colunas obrigatórias faltantes
        if required_columns:
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0  # ou "" dependendo do tipo de coluna

        return df

    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return pd.DataFrame()
