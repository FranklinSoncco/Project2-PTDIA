import time
from pathlib import Path

import streamlit as st

from utils import session as session_utils
from utils import json_manager, inference

BUILD_STEPS = [
    "Analyzing your decisions…",
    "Comparing authenticity scores…",
    "Building explanation…",
]

MODELO_RESUMEN_DIR = Path(__file__).resolve().parent.parent / "modelo_resumen"


def _render_progress(placeholder, label: str, pct: int):
    """Un solo bloque HTML por actualización (a propósito: concatenar
    varios <div> dentro de un mismo st.markdown puede hacer que
    Streamlit pierda el render de los bloques siguientes al primero)."""
    placeholder.markdown(
        f"""
        <div class="dv-glass dv-fade-up" style="padding:30px; text-align:center;">
          <h3 style="font-size:16px; color:#8A93A6; font-weight:500; margin-bottom:18px;">Construyendo el reporte</h3>
          <div class="dv-mono" style="font-size:13.5px; color:var(--text-primary); margin-bottom:12px;">{label}</div>
          <div class="dv-confidence-bar"><span style="width:{pct}%;"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_building():
    """Animación + ejecución real del agente de resumen (parte 4 del
    equipo, o el resumen basado en reglas si aún no llega)."""
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        placeholder = st.empty()

        for label in BUILD_STEPS:
            for pct in (20, 55, 90, 100):
                _render_progress(placeholder, label, pct)
                time.sleep(0.1)

        # build + persist the session JSON, then run the summary agent
        session_json = json_manager.build_session_json(
            session_id=st.session_state.session_id,
            input_image_rel=f"inputs/{Path(st.session_state.uploaded_image_path).name}",
            variations=st.session_state.variations,
        )
        json_path = session_utils.get_subdir("json") / "session.json"
        json_manager.save_json(session_json, json_path)

        summary_text = inference.generate_summary(session_json, model_folder=MODELO_RESUMEN_DIR)

        st.session_state.session_json = session_json
        st.session_state.summary_text = summary_text
        time.sleep(0.2)

    st.session_state.screen = "report"
    st.rerun()


def render_report():
    variations = st.session_state.variations
    accepted = [v for v in variations if v["decision"] == "accepted"]
    rejected = [v for v in variations if v["decision"] == "rejected"]
    confidence = round(sum(v["authenticity_score"] for v in variations) / len(variations) * 100)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.markdown(
            f"""
            <div class="dv-glass dv-fade-up" style="padding:34px;">
              <h2 style="text-align:center;">Audit Summary</h2>
              <p class="dv-mono" style="text-align:center; margin:6px 0 20px; font-size:12px;">
                session · {st.session_state.session_id}
              </p>
              <div class="dv-report-grid">
                <div class="dv-stat"><div class="num">{len(variations)}</div><div class="lbl">Variaciones</div></div>
                <div class="dv-stat"><div class="num" style="color:#5EEAD4;">{len(accepted)}</div><div class="lbl">Aprobadas</div></div>
                <div class="dv-stat"><div class="num" style="color:#F2545B;">{len(rejected)}</div><div class="lbl">Rechazadas</div></div>
              </div>
              <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace; font-size:11.5px; color:#5B6377; margin-bottom:6px;">
                <span>CONFIDENCE</span><span>{confidence}%</span>
              </div>
              <div class="dv-confidence-bar"><span style="width:{confidence}%;"></span></div>
              <div class="dv-explain">{st.session_state.summary_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                "Export JSON",
                data=_json_bytes(),
                file_name=f"{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                "Download Report",
                data=_report_txt(),
                file_name=f"{st.session_state.session_id}_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with b3:
            if st.button("New Audit", type="primary", use_container_width=True):
                session_utils.reset_session(destroy_old=True)
                st.rerun()


def _json_bytes() -> bytes:
    import json as _json
    return _json.dumps(st.session_state.session_json, indent=2, ensure_ascii=False).encode("utf-8")


def _report_txt() -> bytes:
    text = (
        f"DEVERITAS — Audit Summary\n"
        f"Session: {st.session_state.session_id}\n\n"
        f"{st.session_state.summary_text}\n"
    )
    return text.encode("utf-8")
