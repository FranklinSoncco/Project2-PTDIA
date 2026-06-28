import base64
import time
from pathlib import Path

import streamlit as st

from utils import svg


def _img_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    suffix = Path(path).suffix.lower().replace(".", "") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{encoded}"


def render_cards():
    variations = st.session_state.variations
    idx = st.session_state.current_index
    total = len(variations)
    current = variations[idx]

    # arranca el cronómetro de "decision_time_seconds" la primera vez
    # que se muestra esta variación (no se reinicia en reruns incidentales
    # de la misma tarjeta).
    if st.session_state.get("card_shown_idx") != idx:
        st.session_state.card_shown_at = time.time()
        st.session_state.card_shown_idx = idx

    st.markdown(
        f'<div class="dv-fade-up" style="text-align:center; margin-bottom:14px;">'
        f'<div class="dv-eyebrow">Paso 2 de 3 · Variación {idx + 1} / {total}</div></div>',
        unsafe_allow_html=True,
    )

    img_uri = _img_data_uri(current["image_path_abs"])
    dial = svg.dial_svg(current["authenticity_score"], size=40)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <div class="dv-stage">
              <div class="dv-card">
                <img src="{img_uri}">
                <div class="tag">{current['id']}</div>
                <div class="scrim">
                  <div class="dv-score-row">
                    {dial}
                    <div>
                      <div class="dv-score-label">Authenticity score</div>
                      <div class="dv-mono" style="font-size:13px;">{current['authenticity_score']:.2f}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            pass
        with c2:
            b1, b2 = st.columns(2)
            with b1:
                with st.container(key="reject_btn"):
                    if st.button(" ", key="reject_click", help="Rechazar"):
                        _decide("rejected")
            with b2:
                with st.container(key="accept_btn"):
                    if st.button(" ", key="accept_click", help="Aprobar"):
                        _decide("accepted")

        dots = "".join(
            f'<span class="{"done" if i < idx else ("current" if i == idx else "")}"></span>'
            for i in range(total)
        )
        st.markdown(f'<div class="dv-progress-dots">{dots}</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="dv-mono" style="text-align:center; font-size:11.5px; '
            'color:#5B6377; margin-top:14px;">rechazar · aprobar</p>',
            unsafe_allow_html=True,
        )


def _decide(decision: str):
    idx = st.session_state.current_index
    elapsed = round(time.time() - st.session_state.get("card_shown_at", time.time()), 1)
    st.session_state.variations[idx]["decision"] = decision
    st.session_state.variations[idx]["decision_time_seconds"] = elapsed
    st.session_state.current_index += 1

    if st.session_state.current_index >= len(st.session_state.variations):
        st.session_state.screen = "building"
    st.rerun()
