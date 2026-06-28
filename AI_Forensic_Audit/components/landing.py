import streamlit as st
from utils import svg


def render_landing():
    st.markdown(
        f"""
        <div class="dv-hero dv-fade-up">
          <div class="dv-eyebrow">Asistente forense de verificación visual</div>
          <div class="dv-scanring">
            <div class="ring"></div><div class="ring"></div><div class="ring"></div>
            <div class="mark">{svg.logo_svg(70)}</div>
          </div>
          <h1>¿Qué tan real es<br>esta imagen?</h1>
          <p class="dv-sub">
            Cargue un rostro, genere cinco variaciones controladas y revise un score
            de autenticidad para cada una. DEVERITAS documenta su criterio y construye
            el reporte de auditoría.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Start Audit", type="primary", use_container_width=True):
            st.session_state.screen = "upload"
            st.rerun()
