"""
Configuração centralizada do projeto.
Define variáveis de ambiente, paths, constantes e logging.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "core"
DATA_DIR = CORE_DIR / "data"
APP_DIR = PROJECT_ROOT / "app"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models_store"
DB_PATH = DATA_DIR / "ecommerce.db"
SQL_DIR = DATA_DIR / "sql"
CONTEXT_CACHE_DIR = DATA_DIR / "context_cache"

# Criar diretórios se não existirem
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ================== CONSTANTES ==================
REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATORS = 30
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONTEXT_MAX_CHARS = 4000
DEFAULT_TOP_K = 5

# Database tables
DB_TABLES = {
    "SOR": "Source of Record (dados brutos)",
    "SOT": "Source of Truth (dados limpos)",
    "SPEC": "Especificação para modelo (dados agregados)",
}

# ================== OPENAI CONFIG ==================
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
DEFAULT_TEMPERATURE = 0.2

# ================== LOGGING ==================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "chatbot.log"

def setup_logging(
    level: str = LOG_LEVEL,
    log_file: Optional[Path] = LOG_FILE,
    module_name: str = "chatbot",
) -> logging.Logger:
    """
    Configura logging estruturado para o projeto.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho do arquivo de log
        module_name: Nome do módulo para o logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(module_name)
    
    # Não adiciona handlers duplicados
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Formato detalhado
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Handler para arquivo
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Criar logger global
logger = setup_logging()

# ================== VALIDAÇÕES ==================
def validate_config() -> bool:
    """
    Valida configurações críticas do projeto.
    
    Returns:
        True se todas as validações passarem
        
    Raises:
        ValueError: Se alguma validação crítica falhar
    """
    errors = []
    
    # Validar paths
    if not DATA_DIR.exists():
        errors.append(f"Diretório de dados não existe: {DATA_DIR}")
    
    if not SQL_DIR.exists():
        errors.append(f"Diretório SQL não existe: {SQL_DIR}")
    
    # Validar arquivo eCommerce.csv
    ecommerce_csv = DATA_DIR / "eCommerce.csv"
    if not ecommerce_csv.exists():
        errors.append(f"Arquivo eCommerce.csv não encontrado: {ecommerce_csv}")
    
    # Validar OpenAI
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY não configurada. Funcionalidades de chat estarão limitadas.")
    
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Erros de validação encontrados:\n{error_msg}")
        raise ValueError(f"Configuração inválida:\n{error_msg}")
    
    logger.info("✅ Todas as validações passaram")
    return True

# Validar ao importar
try:
    validate_config()
except ValueError as e:
    logger.error(f"Erro ao inicializar config: {e}")
    # Não fazer raise aqui para permitir imports em desenvolvimento

__all__ = [
    "PROJECT_ROOT",
    "CORE_DIR",
    "DATA_DIR",
    "APP_DIR",
    "MODELS_DIR",
    "DB_PATH",
    "SQL_DIR",
    "CONTEXT_CACHE_DIR",
    "REQUIRED_COLUMNS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DEFAULT_TOP_K",
    "DB_TABLES",
    "OPENAI_API_KEY",
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
    "DEFAULT_TEMPERATURE",
    "logger",
    "setup_logging",
    "validate_config",
]
