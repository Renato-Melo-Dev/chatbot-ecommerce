from pathlib import Path
from typing import Optional

from core.config import CONTEXT_CACHE_DIR, logger


def load_context(preferred: Optional[str] = None) -> str:
    """Carrega um arquivo de contexto gerado pelo `generate_context`.

    Args:
        preferred: nome (sem extensão) preferido, ex: 'context_full' ou 'context_describe'.
    Returns:
        Conteúdo do arquivo de contexto.
    """
    cache_dir = Path(CONTEXT_CACHE_DIR)
    if not cache_dir.exists():
        logger.error(f"Diretório de contexto inexistente: {cache_dir}")
        raise FileNotFoundError(f"Diretório de contexto inexistente: {cache_dir}")

    if preferred:
        p = cache_dir / f"{preferred}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")

    # Se nenhum preferido, pega o mais recente context_*.txt
    candidates = sorted(cache_dir.glob("context_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        logger.error("Nenhum arquivo de contexto encontrado em CACHE")
        raise FileNotFoundError("Nenhum arquivo de contexto encontrado em CACHE")

    latest = candidates[0]
    logger.info(f"Carregando contexto: {latest}")
    return latest.read_text(encoding="utf-8")
