"""
components/generator.py
------------------------
Pantalla de generación: muestra la imagen que subió el usuario con una
línea de escaneo animada (CSS puro, no se actualiza desde Python) y una
barra de procesamiento superpuesta sobre la propia imagen, en vez de un
log de texto tipo consola.

Nota técnica: cada actualización se manda en UN SOLO bloque HTML por
llamada a placeholder.markdown(). Concatenar varios <div> dentro de un
mismo st.markdown (como hacía la versión anterior con 3 filas de
"consola") puede hacer que Streamlit renderice el primer bloque bien y
deje los siguientes como texto crudo sin parsear — por eso esta versión
usa un solo contenedor dinámico.
"""

import base64
import time
from pathlib import Path

import streamlit as st

from utils import session as session_utils
from utils import inference

MODELO_GENERATIVO_DIR = Path(__file__).resolve().parent.parent / "modelo_generativo"

PHASE_1 = "Preparing model…"
PHASE_2 = "Generating variations…"
PHASE_3 = "Authenticity analysis…"


def _img_data_uri(path) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = Path(path).suffix.lower().replace(".", "") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{encoded}"


def _render_frame(placeholder, img_uri: str, label: str, pct: int):
    placeholder.markdown(
        f"""
        <div class="dv-fade-up" style="text-align:center;">
          <div class="dv-eyebrow" style="margin-bottom:16px;">Paso 1.5 de 3 · Generando</div>
          <div class="dv-scan-frame">
            <img src="{img_uri}">
            <div class="dv-scan-line"></div>
            <div class="dv-scan-corners"><span></span><span></span><span></span><span></span></div>
            <div class="dv-scan-overlay">
              <div class="dv-scan-status">{label}</div>
              <div class="dv-scan-bar-track"><span style="width:{pct}%;"></span></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_generator():
    img_uri = _img_data_uri(st.session_state.uploaded_image_path)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        placeholder = st.empty()

        # Fase 1 — preparando
        for pct in (10, 35, 60, 100):
            _render_frame(placeholder, img_uri, PHASE_1, pct)
            time.sleep(0.11)

        # Fase 2 — generación real (mock o modelo del equipo, según loader.py)
        _render_frame(placeholder, img_uri, PHASE_2, 12)
        variations = inference.generate_variations(
            input_image_path=Path(st.session_state.uploaded_image_path),
            output_dir=session_utils.get_subdir("output"),
            model_folder=MODELO_GENERATIVO_DIR,
        )
        st.session_state.variations = variations
        for pct in (55, 85, 100):
            _render_frame(placeholder, img_uri, PHASE_2, pct)
            time.sleep(0.11)

        # Fase 3 — scoring de autenticidad
        for pct in (30, 65, 100):
            _render_frame(placeholder, img_uri, PHASE_3, pct)
            time.sleep(0.11)

        time.sleep(0.25)

    st.session_state.current_index = 0
    st.session_state.screen = "evaluate"
    st.rerun()
