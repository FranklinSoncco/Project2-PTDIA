import streamlit as st
from utils import svg


def render_navbar():
    session_id = st.session_state.get("session_id", "—")
    st.markdown(
        f"""
        <div class="dv-nav">
          <div class="dv-nav-brand">{svg.logo_svg(28)}<span>DEVERITAS</span></div>
          <div class="dv-nav-session">
            <span style="color:#5EEAD4;">●</span> session · {session_id}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
