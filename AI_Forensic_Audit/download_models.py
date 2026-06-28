"""
download_models.py
===================
Descarga modelos abiertos para poder probar la interfaz de punta a
punta MIENTRAS el equipo entrena/entrega los modelos definitivos
(parte 2: generativo, parte 4: resumen).

IMPORTANTE — leer antes de correr:

  No existe hoy un modelo único, pre-entrenado, que tome un rostro real
  y devuelva "5 variaciones forenses + score de autenticidad" listo
  para descargar en .safetensors/.pkl/.pth. Lo más cercano es un modelo
  de difusión de propósito general para variaciones de imagen (img2img).
  Este script descarga uno de esos modelos abiertos para que la
  interfaz tenga algo real con qué correr mientras llega el modelo del
  equipo. Si no se corre este script, o si no hay internet/GPU
  disponible, la interfaz sigue funcionando en MODO SIMULADO (ver
  utils/inference.py) — no se rompe nada.

  Este script NO se pudo ejecutar dentro del sandbox de Claude (la red
  de ese entorno solo permite pypi/npm/github, no huggingface.co), así
  que está verificado por sintaxis pero no se corrió la descarga real.
  Debe probarse en un entorno con internet normal (tu laptop, Colab,
  Streamlit Cloud, etc.).

Uso:
    python download_models.py                  # descarga el modelo generativo (recomendado)
    python download_models.py --skip-summary    # (default) no descarga modelo de resumen
    python download_models.py --with-summary    # además descarga un LLM pequeño para el resumen
    python download_models.py --model lambdalabs/sd-image-variations-diffusers

Requiere las dependencias de requirements-models.txt:
    pip install -r requirements-models.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELO_GENERATIVO_DIR = ROOT / "modelo_generativo"
MODELO_RESUMEN_DIR = ROOT / "modelo_resumen"

DEFAULT_GENERATIVE_MODEL = "lambdalabs/sd-image-variations-diffusers"
DEFAULT_SUMMARY_MODEL = "google/flan-t5-small"


def _check_deps():
    missing = []
    for pkg in ("torch", "diffusers", "safetensors", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(
            "Faltan dependencias: " + ", ".join(missing) + "\n"
            "Instala con:  pip install -r requirements-models.txt"
        )
        sys.exit(1)


def download_generative_model(model_id: str, out_dir: Path):
    """Descarga un modelo de difusión open-source capaz de generar
    variaciones de una imagen de entrada (img2img / image-variations),
    y lo guarda como .safetensors en /modelo_generativo.

    Nota de integración: este modelo NO conoce nuestro problema forense
    específico; solo sirve para probar el pipeline completo (carga ->
    inferencia -> 5 salidas -> score) con un modelo real. Cuando el
    equipo entrene/adapte el modelo definitivo, basta con reemplazar el
    archivo .safetensors en esta carpeta y completar
    utils/inference.py -> run_custom_generative_model().
    """
    from diffusers import StableDiffusionImageVariationPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo generativo: {model_id} ...")

    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        model_id, revision="v2.0"
    )

    # Guardamos los pesos del componente principal (UNet) como
    # .safetensors plano en la carpeta acordada con el equipo.
    from safetensors.torch import save_file

    state_dict = pipe.unet.state_dict()
    save_file(state_dict, str(out_dir / "model.safetensors"))
    print(f"Guardado en {out_dir / 'model.safetensors'}")
    print(
        "Nota: este archivo contiene únicamente los pesos del UNet. "
        "Si tu integración necesita el pipeline completo (VAE, image "
        "encoder, scheduler), considera usar pipe.save_pretrained(out_dir) "
        "en su lugar y ajustar utils/loader.py."
    )


def download_summary_model(model_id: str, out_dir: Path):
    """Opcional: descarga un LLM pequeño open-source para generar el
    resumen en lugar del resumen basado en reglas. No es necesario para
    probar la interfaz — utils/inference.py ya tiene un resumen basado
    en reglas que siempre funciona, sin dependencias ni costo."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo de resumen: {model_id} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    tokenizer.save_pretrained(out_dir)
    save_file(model.state_dict(), str(out_dir / "model.safetensors"))
    print(f"Guardado en {out_dir / 'model.safetensors'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_GENERATIVE_MODEL,
                         help="ID de Hugging Face del modelo generativo a descargar")
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL,
                         help="ID de Hugging Face del modelo de resumen (opcional)")
    parser.add_argument("--with-summary", action="store_true",
                         help="También descarga un LLM pequeño para el resumen")
    parser.add_argument("--skip-generative", action="store_true",
                         help="No descargar el modelo generativo")
    args = parser.parse_args()

    _check_deps()

    if not args.skip_generative:
        try:
            download_generative_model(args.model, MODELO_GENERATIVO_DIR)
        except Exception as e:
            print(f"No se pudo descargar el modelo generativo: {e}")
            print("La interfaz seguirá funcionando en modo simulado.")

    if args.with_summary:
        try:
            download_summary_model(args.summary_model, MODELO_RESUMEN_DIR)
        except Exception as e:
            print(f"No se pudo descargar el modelo de resumen: {e}")
            print("Se usará el resumen basado en reglas (utils/inference.py).")

    print("\nListo. Corre la app con:  streamlit run app.py")


if __name__ == "__main__":
    main()
