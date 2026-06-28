"""
test_backend_connection.py
===========================
Prueba DIRECTA contra el backend real (no una réplica) — yo no puedo
llegar a esa URL desde mi entorno (red restringida a pypi/npm/github),
así que esto hay que correrlo desde tu máquina o desde una terminal con
internet normal.

Manda una imagen a /generate, GUARDA las 5 variaciones (+ la
reconstrucción) como archivos .png para que las puedas abrir y revisar,
y muestra exactamente qué campos vienen en la respuesta (sin imprimir
los base64 completos) — así confirmamos si hay algún campo de
métrica/score y cómo se llama exactamente.

Uso:
    pip install requests pillow
    python test_backend_connection.py ruta/a/una/imagen.jpg
"""

import sys
import json
import base64
import io
from pathlib import Path

import requests
from PIL import Image

BACKEND_URL = "https://uptake-pregnant-obstruct.ngrok-free.dev"
HEADERS = {"ngrok-skip-browser-warning": "true"}
OUTPUT_DIR = Path("test_output")


def shorten(d: dict) -> dict:
    """Devuelve una copia del dict sin los valores base64 (muy largos)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and ("b64" in k or len(v) > 200):
            out[k] = f"<string de {len(v)} caracteres, omitido>"
        else:
            out[k] = v
    return out


def save_b64_image(b64_str: str, path: Path):
    img_bytes = base64.b64decode(b64_str)
    Image.open(io.BytesIO(img_bytes)).save(path)


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_backend_connection.py ruta/a/imagen.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Backend: {BACKEND_URL}")
    print(f"Imagen:  {image_path}\n")

    with open(image_path, "rb") as f:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/generate",
                files={"file": (image_path, f, "image/jpeg")},
                headers=HEADERS,
                timeout=120,
            )
        except Exception as e:
            print(f"❌ No se pudo conectar al backend: {e}")
            sys.exit(1)

    print(f"Status code: {resp.status_code}\n")

    try:
        data = resp.json()
    except Exception:
        print("⚠️  La respuesta no es JSON. Primeros 2000 caracteres:")
        print(resp.text[:2000])
        sys.exit(1)

    print("Claves de nivel superior:", list(data.keys()))
    print()

    variations = data.get("variations", [])
    print(f"Cantidad de variaciones: {len(variations)}\n")

    for v in variations:
        print(" -", json.dumps(shorten(v), ensure_ascii=False))

    # ---- guardar las imágenes para verlas localmente -----------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = []

    if data.get("reconstruction_b64"):
        out = OUTPUT_DIR / "reconstruction.png"
        save_b64_image(data["reconstruction_b64"], out)
        saved.append(out)

    for v in variations:
        vid = v.get("id", "V?")
        out = OUTPUT_DIR / f"{vid}.png"
        save_b64_image(v["image_b64"], out)
        saved.append(out)

    print(f"\n🖼️  Imágenes guardadas en: {OUTPUT_DIR.resolve()}")
    for p in saved:
        print("   -", p.name)
    # --------------------------------------------------------------------

    print(
        "\n👉 Revisa si alguna clave de las variaciones es la métrica de "
        "autenticidad (score, confidence, authenticity_score, etc.). Si no "
        "aparece ninguna, hay que pedirle al equipo que la agregue a /generate."
    )


if __name__ == "__main__":
    main()

