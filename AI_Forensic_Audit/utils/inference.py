"""
utils/inference.py
-------------------
Dos responsabilidades:

1. generate_variations(): produce las 5 variaciones visuales + su score
   de autenticidad. Si hay un modelo real en /modelo_generativo, se
   delega a run_custom_generative_model() (que el equipo debe completar
   según la arquitectura final). Si no hay modelo, se usa un generador
   simulado con PIL para poder probar TODA la interfaz de punta a punta
   hoy mismo, sin esperar al modelo del equipo.

2. generate_summary(): el agente de resumen (parte 4 del equipo). Si hay
   un modelo real en /modelo_resumen, se delega a run_custom_summary_model().
   Si no, se usa un resumen basado en reglas (determinístico, sin
   dependencias, nunca falla) que ya combina la decisión humana con el
   score del clasificador, tal como pide el enunciado del proyecto.

Cuando lleguen los modelos reales del equipo, lo único que debería
cambiar en este archivo son los dos `run_custom_*` — el resto del
pipeline (sesión, JSON, limpieza, UI) no se toca.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance

from . import loader

N_VARIATIONS = 5


# --------------------------------------------------------------------------
# Generative step
# --------------------------------------------------------------------------

def run_custom_generative_model(model, image: Image.Image, n: int):
    """Punto de integración para el modelo generativo real (parte 2 del
    equipo). Debe devolver una lista de tuplas (PIL.Image, score_float)
    de longitud `n`, o una lista vacía si no aplica todavía.

    Implementación pendiente: instanciar la arquitectura correspondiente,
    cargar `model` (ya viene como state_dict/tensores desde utils.loader)
    y correr la inferencia real (p. ej. img2img / ControlNet / etc.).
    """
    return []


def _hue_shift(img: Image.Image, shift_amount: int) -> Image.Image:
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda x: (x + shift_amount) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _diff_score(original: Image.Image, variant: Image.Image) -> float:
    """Heurística simple para el modo simulado: a mayor diferencia
    perceptual frente a la imagen original, menor el score de
    autenticidad. Esto es solo para que la demo tenga scores con algo
    de lógica visible; el modelo real definirá su propio criterio."""
    a = original.convert("RGB").resize((64, 64))
    b = variant.convert("RGB").resize((64, 64))
    pa, pb = list(a.getdata()), list(b.getdata())
    diff = sum(abs(p1[c] - p2[c]) for p1, p2 in zip(pa, pb) for c in range(3))
    norm = diff / (64 * 64 * 3 * 255)
    score = 1 - norm * 1.8 + random.uniform(-0.05, 0.05)
    return round(max(0.05, min(0.97, score)), 4)


def _mock_variations(original: Image.Image, n: int):
    """Genera `n` variaciones visualmente distintas con transformaciones
    deterministas de PIL (no usa ningún modelo de IA). Suficiente para
    probar carga de imagen -> tarjetas -> decisiones -> reporte sin
    depender de pesos que aún no existen."""
    recipes = [
        lambda im: ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.08)).enhance(1.35),
        lambda im: ImageEnhance.Brightness(_hue_shift(im, 18)).enhance(1.06),
        lambda im: ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(1.04)).enhance(0.55),
        lambda im: ImageEnhance.Color(_hue_shift(im, 235)).enhance(0.85),
        lambda im: ImageEnhance.Sharpness(ImageEnhance.Contrast(im).enhance(1.22)).enhance(1.6),
    ]
    out = []
    for i in range(n):
        recipe = recipes[i % len(recipes)]
        variant = recipe(original.copy())
        score = _diff_score(original, variant)
        out.append((variant, score))
    return out


def generate_variations(
    input_image_path: Path,
    output_dir: Path,
    model_folder: Optional[Path] = None,
    n: int = N_VARIATIONS,
) -> list[dict]:
    original = Image.open(input_image_path).convert("RGB")

    model = loader.load_model(model_folder) if model_folder else None
    results = run_custom_generative_model(model, original, n) if model is not None else []
    if not results:
        results = _mock_variations(original, n)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, (img, score) in enumerate(results, start=1):
        vid = f"V{i}"
        out_path = output_dir / f"var_{i}.png"
        img.save(out_path)
        saved.append(
            {
                "id": vid,
                "image_path_abs": str(out_path),
                "image_path": f"outputs/var_{i}.png",
                "authenticity_score": score,
                "decision": None,
            }
        )
    return saved


# --------------------------------------------------------------------------
# Summary / explanation agent
# --------------------------------------------------------------------------

def run_custom_summary_model(model, session_json: dict) -> str:
    """Punto de integración para el agente de resumen real (parte 4 del
    equipo, basado en LLM). Debe devolver el texto del resumen, o ''
    si no aplica todavía."""
    return ""


def _rule_based_summary(session_json: dict) -> str:
    variations = session_json.get("variations", [])
    accepted = [v for v in variations if v.get("decision") == "accepted"]
    rejected = [v for v in variations if v.get("decision") == "rejected"]

    avg_acc = sum(v["authenticity_score"] for v in accepted) / len(accepted) if accepted else 0
    avg_rej = sum(v["authenticity_score"] for v in rejected) / len(rejected) if rejected else 0

    lines = [
        f"De las {len(variations)} variaciones generadas, el usuario aprobó "
        f"{len(accepted)} y rechazó {len(rejected)}."
    ]

    if accepted:
        ids = ", ".join(v["id"] for v in accepted)
        coherente = avg_acc >= 0.6
        lines.append(
            f"Las variaciones aprobadas ({ids}) registraron un score de autenticidad "
            f"promedio de {avg_acc:.2f}, "
            + (
                "consistente con un criterio alineado al del clasificador forense."
                if coherente
                else "lo cual sugiere un criterio más permisivo que el del clasificador en al menos un caso."
            )
        )

    if rejected:
        ids = ", ".join(v["id"] for v in rejected)
        coherente = avg_rej < 0.5
        lines.append(
            f"Las variaciones rechazadas ({ids}) promediaron un score de {avg_rej:.2f}, "
            + (
                "coincidiendo con señales de manipulación detectadas automáticamente."
                if coherente
                else "a pesar de un score relativamente alto, lo que indica un criterio más estricto por parte del usuario."
            )
        )

    lines.append(
        "Este resumen combina la decisión humana con la señal del modelo generativo "
        "y se conserva únicamente durante la sesión activa."
    )
    return "\n\n".join(lines)


def generate_summary(session_json: dict, model_folder: Optional[Path] = None) -> str:
    model = loader.load_model(model_folder) if model_folder else None
    if model is not None:
        text = run_custom_summary_model(model, session_json)
        if text:
            return text
    return _rule_based_summary(session_json)
