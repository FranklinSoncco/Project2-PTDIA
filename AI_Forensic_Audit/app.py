"""
DEVERITAS — Asistente forense de verificación visual
======================================================
Proyecto #2 (Planificación y Toma de Decisiones en IA, UTEC) — parte 3:
interfaz de interacción humano-IA.

Flujo: Landing -> Drag & Drop -> Generando -> Evaluación (cards) ->
Construyendo reporte -> Reporte -> Nueva auditoría.

Cada cliente conectado recibe su propia carpeta de sesión (ver
utils/session.py) por lo que varios usuarios pueden usar la app al
mismo tiempo sin que sus imágenes se mezclen. Al terminar una
auditoría (o purgar sesiones viejas) los archivos se destruyen.
"""

from pathlib import Path

import streamlit as st
from PIL import Image

from utils import session as session_utils
from utils import cleanup, styling
from components import navbar, landing, uploader, generator, cards, report

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="DEVERITAS",
    page_icon=Image.open(ROOT / "assets" / "favicon.png"),
    layout="centered",
)

session_id = session_utils.init_session()
cleanup.purge_stale_sessions(max_age_minutes=30, keep=session_id)

styling.inject_global_css()

navbar.render_navbar()

screen = st.session_state.screen

if screen == "landing":
    landing.render_landing()
elif screen == "upload":
    uploader.render_uploader()
elif screen == "generating":
    generator.render_generator()
elif screen == "evaluate":
    cards.render_cards()
elif screen == "building":
    report.render_building()
elif screen == "report":
    report.render_report()
else:
    st.session_state.screen = "landing"
    st.rerun()
