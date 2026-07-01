import time
from pathlib import Path

import streamlit as st

from utils import session as session_utils
from utils import json_manager, inference, api_client

BUILD_STEPS = [
    "Analyzing your decisions…",
    "Comparing authenticity scores…",
    "Building explanation…",
]

MODELO_RESUMEN_DIR = Path(__file__).resolve().parent.parent / "modelo_resumen"

SUMMARY_BADGES = {
    "agent": ("#5EEAD4", "resumen vía agente (Gemini)"),
    "rule_based": ("#F5B95B", "resumen basado en reglas (sin agente)"),
}


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
    """Animación + ejecución real del agente de resumen (AuditAgent /
    Gemini si hay GEMINI_API_KEY, si no resumen basado en reglas)."""
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        placeholder = st.empty()

        for label in BUILD_STEPS:
            for pct in (20, 55, 90, 100):
                _render_progress(placeholder, label, pct)
                time.sleep(0.1)

        # build + persist the session JSON (formato exacto que pide el agente),
        # luego correr el agente de resumen
        session_json = json_manager.build_session_json(
            session_id=st.session_state.session_id,
            input_image_rel=f"inputs/{Path(st.session_state.uploaded_image_path).name}",
            reconstruction_b64=st.session_state.reconstruction_b64,
            variations=st.session_state.variations,
        )
        json_path = session_utils.get_subdir("json") / "session.json"
        json_manager.save_json(session_json, json_path)

        summary_result = inference.generate_summary(session_json, model_folder=MODELO_RESUMEN_DIR)

        st.session_state.session_json = session_json
        st.session_state.summary_result = summary_result

        # envío "best effort" al backend del equipo — no bloquea ni rompe
        # el reporte si falla, porque el resumen ya se construyó localmente
        backend_url = api_client.get_backend_url()
        if api_client.backend_is_configured(backend_url):
            decisions = [
                {
                    "id": v["id"],
                    "accepted": v["decision"] == "accepted",
                    "authenticity_score": v["authenticity_score"],
                }
                for v in st.session_state.variations
            ]
            # usar el session_id que devolvió /generate (el de ellos), no el nuestro
            feedback_session_id = st.session_state.get("backend_session_id") or st.session_state.session_id
            api_client.send_feedback(backend_url, feedback_session_id, decisions)

        time.sleep(0.2)

    st.session_state.screen = "report"
    st.rerun()


def render_report():
    variations = st.session_state.variations
    accepted = [v for v in variations if v["decision"] == "accepted"]
    rejected = [v for v in variations if v["decision"] == "rejected"]
    confidence = round(sum(v["authenticity_score"] for v in variations) / len(variations) * 100)
    summary_result = st.session_state.summary_result
    color, badge_label = SUMMARY_BADGES[summary_result["source"]]

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.container(key="report_card"):
            st.markdown(
                f"""
                <h2 style="text-align:center;">Audit Summary</h2>
                <p class="dv-mono" style="text-align:center; margin:6px 0 4px; font-size:12px;">
                  session · {st.session_state.session_id}
                </p>
                <p style="text-align:center; margin-bottom:20px;">
                  <span class="dv-mono" style="color:{color}; border:1px solid {color}55;
                    padding:2px 10px; border-radius:999px; font-size:11px;">{badge_label}</span>
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
                <div style="margin-top:18px;"></div>
                """,
                unsafe_allow_html=True,
            )

            if summary_result["source"] == "agent":
                # texto real de Gemini con ## headers — se renderiza con
                # markdown nativo de Streamlit para que se vea con formato,
                # no como texto plano con '##' literal.
                st.markdown(summary_result["text"])
            else:
                st.markdown(
                    f'<div class="dv-explain">{summary_result["text"]}</div>',
                    unsafe_allow_html=True,
                )
                if summary_result.get("error_detail"):
                    with st.expander("¿Por qué no se usó el agente (Gemini)?"):
                        st.code(summary_result["error_detail"], language=None)

        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                "Export JSON",
                data=_export_json_bytes(),
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


def _export_json_bytes() -> bytes:
    import json as _json
    data = json_manager.build_export_json(st.session_state.session_id, st.session_state.variations)
    return _json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def _report_txt() -> bytes:
    text = (
        f"DEVERITAS — Audit Summary\n"
        f"Session: {st.session_state.session_id}\n\n"
        f"{st.session_state.summary_result['text']}\n"
    )
    return text.encode("utf-8")
