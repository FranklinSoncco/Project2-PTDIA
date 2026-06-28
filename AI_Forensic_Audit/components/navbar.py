import streamlit as st
from utils import svg

SOURCE_BADGES = {
    "backend": ("#5EEAD4", "fuente: backend"),
    "local_model": ("#8B7CF6", "fuente: modelo local"),
    "mock": ("#F5B95B", "fuente: simulado (sin backend)"),
}


def render_navbar():
    session_id = st.session_state.get("session_id", "—")
    source = st.session_state.get("generation_source")

    badge_html = ""
    if source in SOURCE_BADGES:
        color, label = SOURCE_BADGES[source]
        badge_html = (
            f'<span class="dv-mono" style="color:{color}; margin-left:14px; '
            f'border:1px solid {color}55; padding:2px 9px; border-radius:999px; '
            f'font-size:11px;">{label}</span>'
        )

    st.markdown(
        f"""
        <div class="dv-nav">
          <div class="dv-nav-brand">{svg.logo_svg(28)}<span>DEVERITAS</span></div>
          <div class="dv-nav-session">
            <span style="color:#5EEAD4;">●</span> session · {session_id}{badge_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
