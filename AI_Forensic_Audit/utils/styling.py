"""
utils/styling.py
-----------------
Inyecta el CSS una sola vez por script-run: el design system compartido
(assets/css/styles.css, el mismo que usa el prototipo HTML) + overrides
específicos de Streamlit para que los widgets nativos (file_uploader,
button, etc.) se vean parte del mismo sistema visual en vez del Streamlit
por defecto.
"""

from pathlib import Path
import base64
import streamlit as st

from utils import svg as svg_module

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def _icon_data_uri(svg_text: str) -> str:
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


STREAMLIT_OVERRIDES_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

#MainMenu, header, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 760px; }
body, .stApp {
  background: var(--bg);
  background-image:
    radial-gradient(circle at 18% 8%, rgba(139,124,246,0.10), transparent 40%),
    radial-gradient(circle at 85% 18%, rgba(94,234,212,0.07), transparent 45%);
  color: var(--text-primary);
}
.stApp::before{
  content:"";
  position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events:none;
  mask-image: radial-gradient(circle at 50% 0%, rgba(0,0,0,0.6), transparent 70%);
  z-index: 0;
}

/* file uploader -> looks like the custom dropzone */
[data-testid="stFileUploaderDropzone"]{
  background: var(--glass) !important;
  border: 1.5px dashed var(--border-strong) !important;
  border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploaderDropzone"] section{ background: transparent !important; }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{
  color: var(--text-tertiary) !important;
}

/* primary CTA buttons (Start Audit, New Audit) */
.stButton button[kind="primary"]{
  background: var(--grad-verify) !important;
  color: #06110F !important;
  border: none !important;
  border-radius: var(--radius-pill) !important;
  font-weight: 600 !important;
  padding: 0.7rem 1.6rem !important;
  box-shadow: 0 12px 28px rgba(94,234,212,0.16) !important;
}
.stButton button[kind="secondary"]{
  background-color: transparent !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-pill) !important;
  color: var(--text-primary) !important;
}

/* custom circular accept / reject buttons, targeted via st.container(key=...) */
.st-key-accept_btn button, .st-key-reject_btn button{
  width: 58px !important; height: 58px !important; border-radius: 50% !important;
  font-size: 0 !important; padding: 0 !important;
  background-color: var(--glass) !important;
  border: 1px solid var(--border-strong) !important;
  background-repeat: no-repeat !important;
  background-position: center !important;
  background-size: 24px 24px !important;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.st-key-accept_btn button{ background-image: url("__ACCEPT_ICON__") !important; }
.st-key-reject_btn button{ background-image: url("__REJECT_ICON__") !important; }
.st-key-accept_btn button:hover{ border-color: var(--cyan) !important; box-shadow: 0 8px 24px rgba(94,234,212,0.25) !important; transform: translateY(-2px); }
.st-key-reject_btn button:hover{ border-color: var(--danger) !important; box-shadow: 0 8px 24px rgba(242,84,91,0.25) !important; transform: translateY(-2px); }

.st-key-report_card{
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 34px !important;
  box-shadow: var(--shadow-glass) !important;
}
.st-key-report_card [data-testid="stMarkdownContainer"] h2{ font-size: 22px; }
.st-key-report_card [data-testid="stMarkdownContainer"] strong{ color: var(--text-primary); }

[data-testid="stMarkdownContainer"] p { margin-bottom: 0; }
"""


def inject_global_css():
    """Streamlit reconstruye el árbol de elementos en cada rerun (no es
    incremental a nivel de script), así que el <style> debe emitirse en
    TODAS las corridas — cachear "ya se inyectó" haría que desaparezca
    en el segundo rerun."""
    base_css = (ASSETS_DIR / "css" / "styles.css").read_text(encoding="utf-8")
    overrides = (
        STREAMLIT_OVERRIDES_TEMPLATE
        .replace("__ACCEPT_ICON__", _icon_data_uri(svg_module.ACCEPT_ICON))
        .replace("__REJECT_ICON__", _icon_data_uri(svg_module.REJECT_ICON))
    )
    st.markdown(f"<style>{base_css}\n{overrides}</style>", unsafe_allow_html=True)
