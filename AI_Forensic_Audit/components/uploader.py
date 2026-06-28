import streamlit as st
from PIL import Image

from utils import session as session_utils


def render_uploader():
    st.markdown(
        """
        <div class="dv-fade-up" style="text-align:center; margin-bottom:10px;">
          <div class="dv-eyebrow">Paso 1 de 3</div>
          <h2 style="font-size:24px; margin:8px 0 4px;">Cargar imagen de entrada</h2>
          <p>Un rostro real, de frente, con buena iluminación. Acepta .jpg, .jpeg, .png</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        uploaded = st.file_uploader(
            "Arrastre la imagen aquí o haga clic para seleccionar",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    st.markdown(
        '<p class="dv-mono" style="text-align:center; font-size:12px; color:#5B6377;">'
        "inputs/ · se descarta al finalizar la sesión</p>",
        unsafe_allow_html=True,
    )

    if uploaded is not None:
        input_dir = session_utils.get_subdir("input")
        ext = uploaded.name.split(".")[-1].lower()
        dest = input_dir / f"source.{ext}"

        image = Image.open(uploaded).convert("RGB")
        image.save(dest)

        st.session_state.uploaded_image_path = str(dest)
        st.session_state.screen = "generating"
        st.rerun()
