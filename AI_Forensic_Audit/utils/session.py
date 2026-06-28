"""
utils/session.py
-----------------
Cada cliente conectado a la app de Streamlit ya corre en su propio
"script run" con su propio `st.session_state` (Streamlit aísla el estado
de cada sesión automáticamente). Lo que SÍ se comparte entre usuarios es
el disco, así que cada sesión recibe su propia carpeta con UUID bajo
`sessions/<session_id>/` para que dos personas usando la app al mismo
tiempo (PEPE1, PEPE2, ...) nunca lean o sobrescriban los archivos del otro.

Estructura por sesión:
    sessions/<session_id>/
        input/   -> imagen original subida por el usuario
        output/  -> 5 variaciones generadas
        json/    -> session JSON con scores + decisiones
        logs/    -> trazas simples de la corrida (debug)
"""

from __future__ import annotations
import uuid
import time
from pathlib import Path

import streamlit as st

BASE_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"
SUBFOLDERS = ("input", "output", "json", "logs")


def _new_session_id() -> str:
    return f"audit_{uuid.uuid4().hex[:10]}"


def init_session() -> str:
    """Crea (si no existe) el estado y las carpetas de la sesión actual.
    Devuelve el session_id. Idempotente: si ya existe, no hace nada.
    """
    if "session_id" not in st.session_state:
        session_id = _new_session_id()
        st.session_state.session_id = session_id
        st.session_state.session_created_at = time.time()
        st.session_state.screen = "landing"
        st.session_state.uploaded_image_path = None
        st.session_state.variations = []
        st.session_state.current_index = 0
        st.session_state.report = None
        st.session_state.backend_session_id = None
        st.session_state.generation_source = None
        st.session_state.reconstruction_b64 = None
        st.session_state.summary_result = None
        _create_dirs(session_id)
    return st.session_state.session_id


def _create_dirs(session_id: str) -> Path:
    session_dir = BASE_SESSIONS_DIR / session_id
    for sub in SUBFOLDERS:
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    return session_dir


def get_session_dir() -> Path:
    init_session()
    return BASE_SESSIONS_DIR / st.session_state.session_id


def get_subdir(name: str) -> Path:
    """name in {'input','output','json','logs'}"""
    assert name in SUBFOLDERS, f"Carpeta de sesión inválida: {name}"
    d = get_session_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def reset_session(destroy_old: bool = True):
    """Inicia una nueva auditoría. Por diseño, todo lo de la sesión
    anterior se destruye (no debe quedar memoria de la imagen, las
    variaciones ni el JSON una vez termina el flujo)."""
    from . import cleanup as _cleanup

    if destroy_old and "session_id" in st.session_state:
        _cleanup.cleanup_session(BASE_SESSIONS_DIR / st.session_state.session_id)

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    init_session()
