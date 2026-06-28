"""
utils/svg.py
------------
SVGs propios (sin stickers, sin emojis) reutilizados por los distintos
componentes de la interfaz: el logo, los íconos de aprobar/rechazar
diseñados a medida, y el dial de score de autenticidad.
"""

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


import re


def _flatten(svg_text: str) -> str:
    """Streamlit's markdown->HTML pass-through can mis-parse raw SVG when an
    attribute value spans multiple lines (e.g. a multi-line `d="...".
    Collapsing all whitespace to single spaces keeps the SVG valid while
    guaranteeing every tag stays on one line."""
    return re.sub(r"\s+", " ", svg_text).strip()


def logo_svg(size: int = 30) -> str:
    raw = (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
    raw = raw.replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)
    return _flatten(raw)


ACCEPT_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#5EEAD4" stroke-width="1.8">
  <circle cx="12" cy="12" r="8.4"/>
  <path d="M8.4 12.4l2.4 2.4 4.8-5.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""".strip()

REJECT_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#F2545B" stroke-width="1.8">
  <circle cx="12" cy="12" r="8.4"/>
  <path d="M8.2 15.8L15.8 8.2" stroke-linecap="round"/>
</svg>
""".strip()

UPLOAD_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#8A93A6" stroke-width="1.6">
  <path d="M12 16V4M12 4l-4 4M12 4l4 4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
""".strip()


def score_color(score: float) -> str:
    if score < 0.45:
        return "#F2545B"
    if score < 0.7:
        return "#F5B95B"
    return "#5EEAD4"


def dial_svg(score: float, size: int = 40) -> str:
    color = score_color(score)
    r = 15
    circ = 2 * 3.14159265 * r
    offset = circ * (1 - score)
    raw = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 38 38">
      <circle cx="19" cy="19" r="{r}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="3.4"/>
      <circle cx="19" cy="19" r="{r}" fill="none" stroke="{color}" stroke-width="3.4"
        stroke-linecap="round" stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
        transform="rotate(-90 19 19)"/>
      <text x="19" y="22.5" text-anchor="middle" font-family="JetBrains Mono, monospace"
        font-size="10" font-weight="600" fill="#E7EAF0">{round(score*100)}</text>
    </svg>
    """
    return _flatten(raw)
