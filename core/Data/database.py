import os
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from core.config import DB_PATH, SQL_DIR, logger

# Diretório base deste módulo
BASE_DIR = Path(__file__).parent

# Engine SQLAlchemy (lazy)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(f"sqlite:///{DB_PATH}")
            logger.info(f"✅ Engine criada em: {DB_PATH}")
        except Exception:
            logger.exception("Erro ao criar engine SQLAlchemy")
            raise
    return _engine


engine = get_engine()


def create_database_and_tables() -> None:
    """Cria o banco e executa arquivos SQL encontrados em `core/data/sql/`.

    Lança `FileNotFoundError` se algum arquivo SQL listado não existir.
    """
    os.makedirs(BASE_DIR, exist_ok=True)

    sql_files = [
        "sor_sales.sql",
        "sot_sales.sql",
        "spec_sales.sql",
    ]

    logger.info("Iniciando criação do banco e execução dos scripts SQL")
    with engine.begin() as conn:
        for sql_file in sql_files:
            file_path = Path(SQL_DIR) / sql_file
            if not file_path.is_file():
                logger.error(f"Arquivo SQL não encontrado: {file_path}")
                raise FileNotFoundError(f"Arquivo SQL não encontrado: {file_path}")

            logger.debug(f"Lendo SQL: {file_path}")
            content = file_path.read_text(encoding="utf-8")
            queries = [q.strip() for q in content.split(";") if q.strip()]
            for q in queries:
                try:
                    conn.execute(text(q))
                except Exception:
                    logger.exception(f"Erro executando query do arquivo {sql_file}")
                    raise

        # Tentar limpar tabelas (se existirem)
        for table in ["SOR", "SOT", "spec_sales"]:
            try:
                conn.execute(text(f"DELETE FROM {table}"))
                logger.debug(f"Limpeza: tabela {table} (DELETE)")
            except Exception:
                logger.debug(f"Tabela {table} não existe ou não pôde ser limpa")

    logger.info("Banco de dados criado/atualizado com sucesso")


def insert_csv_to_sor(df: pd.DataFrame) -> int:
    """Insere um DataFrame na tabela `SOR`.

    Retorna o número de linhas inseridas.
    """
    if df is None or df.empty:
        logger.error("DataFrame vazio passado para insert_csv_to_sor")
        raise ValueError("DataFrame não pode ser vazio")
    try:
        n = df.shape[0]
        df.to_sql("SOR", engine, if_exists="replace", index=False)
        logger.info(f"{n} linhas inseridas em SOR")
        return n
    except Exception:
        logger.exception("Erro ao inserir CSV em SOR")
        raise


def run_etl_sor_to_sot() -> None:
    """Transforma dados de `SOR` para `SOT` aplicando filtros básicos."""
    logger.info("Iniciando ETL: SOR -> SOT")
    query = (
        "SELECT InvoiceNo, StockCode, Description, Quantity,"
        " UnitPrice, CustomerID, Country, InvoiceDate,"
        " (Quantity * UnitPrice) AS TotalPrice"
        " FROM SOR"
        " WHERE Quantity > 0 AND UnitPrice > 0 AND CustomerID IS NOT NULL"
    )
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            logger.warning("Nenhuma linha válida encontrada em SOR para transformação")
        else:
            logger.info(f"Encontradas {len(df)} linhas válidas em SOR")
        df.to_sql("SOT", engine, if_exists="replace", index=False)
        logger.info("ETL SOR -> SOT concluído")
    except Exception:
        logger.exception("Falha no ETL SOR -> SOT")
        raise


def run_etl_sot_to_spec_train() -> None:
    """Agrega `SOT` para gerar `spec_sales` usada no treino."""
    logger.info("Iniciando ETL: SOT -> spec_sales (treino)")
    query = (
        "SELECT CustomerID, Country, SUM(TotalPrice) AS TotalPrice"
        " FROM SOT GROUP BY CustomerID, Country"
    )
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Agrupamentos gerados: {len(df)}")
        df.to_sql("spec_sales", engine, if_exists="replace", index=False)
        logger.info("spec_sales criada com sucesso")
    except Exception:
        logger.exception("Falha no ETL SOT -> spec_sales")
        raise


def run_etl_for_test_data(df_test: pd.DataFrame) -> None:
    """Prepara e grava `spec_sales` com dados de teste agregados."""
    if df_test is None or df_test.empty:
        logger.error("df_test vazio em run_etl_for_test_data")
        raise ValueError("df_test não pode ser vazio")
    try:
        df_test = df_test.copy()
        df_test["TotalPrice"] = df_test.get("Quantity", 0) * df_test.get("UnitPrice", 0)
        df = df_test.groupby(["CustomerID", "Country"], as_index=False)["TotalPrice"].sum()
        df.to_sql("spec_sales", engine, if_exists="replace", index=False)
        logger.info("spec_sales (teste) preparada com sucesso")
    except Exception:
        logger.exception("Erro ao preparar dados de teste para spec_sales")
        raise


def load_data(table_name: str) -> pd.DataFrame:
    """Carrega todos os dados de uma tabela do DB e retorna um DataFrame."""
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        logger.info(f"Carregadas {len(df)} linhas de {table_name}")
        return df
    except Exception:
        logger.exception(f"Erro ao carregar dados da tabela {table_name}")
        raise
