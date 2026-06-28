"""
utils/cleanup.py
----------------
"Una vez que termine el proceso ya sea de PEPE1, PEPE2 o cualquier
usuario, toda la data se destruye para no ocupar memoria."

Dos mecanismos de limpieza:
1. cleanup_session(path)         -> borrado explícito (botón "Nueva Auditoría",
                                     o el usuario termina su auditoría).
2. purge_stale_sessions(max_age) -> red de seguridad para sesiones abandonadas
                                     (el usuario cierra la pestaña sin terminar).
                                     Se llama de forma oportunista al inicio
                                     de cada script run en app.py.
"""

from __future__ import annotations
import shutil
import time
from pathlib import Path

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


def cleanup_session(session_dir: Path) -> None:
    """Elimina por completo la carpeta de una sesión (imagen de entrada,
    variaciones generadas, JSON y logs). No debe lanzar excepción aunque
    la carpeta ya no exista o esté parcialmente borrada."""
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass


def purge_stale_sessions(max_age_minutes: int = 30, keep: str | None = None) -> int:
    """Recorre sessions/ y elimina cualquier carpeta más vieja que
    `max_age_minutes`, excepto `keep` (el session_id activo). Devuelve
    cuántas carpetas se eliminaron. Es seguro llamarla en cada rerun:
    es barata si no hay nada que limpiar.
    """
    if not SESSIONS_DIR.exists():
        return 0

    removed = 0
    cutoff = time.time() - max_age_minutes * 60
    for child in SESSIONS_DIR.iterdir():
        if not child.is_dir():
            continue
        if keep and child.name == keep:
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except FileNotFoundError:
            continue
    return removed
