"""
utils/inference.py
-------------------
Dos responsabilidades:

1. generate_variations(): produce las 5 variaciones visuales (imagen +
   label + description) más reconstruction_b64. Prioridad: backend
   externo -> modelo local en /modelo_generativo -> modo simulado.
   Nunca deja la interfaz sin variaciones. Las 4 métricas (MAD, FID,
   LPIPS, ArcFace) se calculan SIEMPRE de forma local en este módulo
   (utils/metrics.py), sin importar de dónde vinieron las imágenes —
   tal como se decidió: "todas las métricas las implementamos de
   manera local para mostrarlo en la interfaz".

2. generate_summary(): el agente de resumen (parte 4 del equipo).
   Prioridad: AuditAgent real (Gemini, ver utils/audit_agent.py) si hay
   GEMINI_API_KEY configurada -> resumen basado en reglas (siempre
   funciona, sin dependencias) si el agente real falla o no está
   configurado.
"""

from __future__ import annotations
import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance

from . import loader
from . import api_client
from . import audit_agent
from . import metrics

N_VARIATIONS = 5


# --------------------------------------------------------------------------
# Generative step
# --------------------------------------------------------------------------

def run_custom_generative_model(model, image: Image.Image, n: int):
    """Punto de integración para el modelo generativo real si se entrega
    como archivo de pesos en /modelo_generativo (alternativa al backend
    HTTP). Debe devolver una lista de `n` PIL.Image, o [] si no aplica
    todavía. (Las métricas ya NO se reciben de aquí: se calculan
    siempre localmente en utils/metrics.py.)
    """
    return []


def _hue_shift(img: Image.Image, shift_amount: int) -> Image.Image:
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda x: (x + shift_amount) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


MOCK_RECIPES = [
    ("Mayor saturación", "Variación simulada con mayor saturación de color.",
     lambda im: ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.08)).enhance(1.35)),
    ("Tono más cálido", "Variación simulada con una iluminación más cálida.",
     lambda im: ImageEnhance.Brightness(_hue_shift(im, 18)).enhance(1.06)),
    ("Menor saturación", "Variación simulada con tono más apagado, estilo sepia.",
     lambda im: ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(1.04)).enhance(0.55)),
    ("Tono más frío", "Variación simulada con una iluminación más fría.",
     lambda im: ImageEnhance.Color(_hue_shift(im, 235)).enhance(0.85)),
    ("Mayor contraste", "Variación simulada con mayor contraste y nitidez.",
     lambda im: ImageEnhance.Sharpness(ImageEnhance.Contrast(im).enhance(1.22)).enhance(1.6)),
]


def _mock_variations(original: Image.Image, n: int):
    """Genera `n` variaciones con transformaciones deterministas de PIL
    (sin ningún modelo de IA). label/description quedan parejos con la
    transformación real aplicada, para que el agente de resumen tenga
    algo coherente con qué trabajar incluso en modo simulado."""
    out = []
    for i in range(n):
        label, description, recipe = MOCK_RECIPES[i % len(MOCK_RECIPES)]
        variant = recipe(original.copy())
        out.append({"image": variant, "label": label, "description": description})
    return out


def generate_variations(
    input_image_path: Path,
    output_dir: Path,
    model_folder: Optional[Path] = None,
    n: int = N_VARIATIONS,
    backend_url: Optional[str] = None,
) -> dict:
    """Devuelve:
        {
          "variations": [ {id, image_path_abs, image_path, image_b64,
                            label, description, authenticity_score,
                            fid, lpips, arcface,
                            decision, decision_time_seconds}, ... ],
          "backend_session_id": str | None,
          "source": "backend" | "local_model" | "mock",
          "reconstruction_b64": str,
        }
    `backend_session_id` es el que devolvió el backend externo — hay
    que reenviarlo tal cual en /feedback. `reconstruction_b64` se
    sintetiza localmente a partir de la imagen de entrada.

    Las 4 métricas (authenticity_score=MAD, fid, lpips, arcface) se
    calculan SIEMPRE aquí mismo, vía utils/metrics.py, sin importar si
    las imágenes vinieron del backend, de un modelo local o del modo
    simulado.
    """
    original = Image.open(input_image_path).convert("RGB")

    backend_session_id = None
    items = []
    source = "mock"

    # 1) backend externo del equipo — prioridad si está configurado
    if api_client.backend_is_configured(backend_url):
        backend_result = api_client.call_generative_backend(backend_url, original, n)
        if backend_result and backend_result["variations"]:
            backend_session_id = backend_result["backend_session_id"]
            source = "backend"
            for item in backend_result["variations"]:
                meta = item.get("meta") or {}
                items.append({
                    "image": item["image"],
                    "label": meta.get("label") or "Variación",
                    "description": meta.get("description") or "",
                })

    # 2) modelo local en /modelo_generativo, si no hubo backend o falló
    if not items and model_folder:
        model = loader.load_model(model_folder)
        if model is not None:
            local_images = run_custom_generative_model(model, original, n)
            if local_images:
                source = "local_model"
                for img in local_images:
                    items.append({"image": img, "label": "Variación", "description": ""})

    # 3) modo simulado — nunca deja la interfaz sin variaciones
    if not items:
        source = "mock"
        items = _mock_variations(original, n)

    # --- las 4 métricas, siempre calculadas localmente -----------------
    metric_results = metrics.compute_all_metrics(original, [it["image"] for it in items])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, (item, m) in enumerate(zip(items, metric_results), start=1):
        vid = f"V{i}"
        out_path = output_dir / f"var_{i}.png"
        item["image"].save(out_path)
        image_b64 = base64.b64encode(out_path.read_bytes()).decode("ascii")
        saved.append(
            {
                "id": vid,
                "image_path_abs": str(out_path),
                "image_path": f"outputs/var_{i}.png",
                "image_b64": image_b64,
                "label": item["label"],
                "description": item["description"],
                "authenticity_score": m["MAD"],
                "fid": m["FID"],
                "lpips": m["LPIPS"],
                "arcface": m["ArcFace"],
                "decision": None,
                "decision_time_seconds": None,
            }
        )

    recon_buf = io.BytesIO()
    original.save(recon_buf, format="PNG")
    reconstruction_b64 = base64.b64encode(recon_buf.getvalue()).decode("ascii")

    return {
        "variations": saved,
        "backend_session_id": backend_session_id,
        "source": source,
        "reconstruction_b64": reconstruction_b64,
    }


# --------------------------------------------------------------------------
# Summary / explanation agent
# --------------------------------------------------------------------------

def _rule_based_summary(session_json: dict) -> str:
    variations = session_json.get("variations", [])
    accepted = [v for v in variations if v.get("decision") == "accepted"]
    rejected = [v for v in variations if v.get("decision") == "rejected"]

    avg_acc = sum(v["MAD"] for v in accepted) / len(accepted) if accepted else 0
    avg_rej = sum(v["MAD"] for v in rejected) / len(rejected) if rejected else 0

    lines = [
        f"De las {len(variations)} variaciones generadas, el usuario aprobó "
        f"{len(accepted)} y rechazó {len(rejected)}."
    ]

    if accepted:
        ids = ", ".join(v["id"] for v in accepted)
        coherente = avg_acc >= 0.6
        lines.append(
            f"Las variaciones aprobadas ({ids}) registraron un MAD promedio "
            f"de {avg_acc:.2f}, "
            + (
                "consistente con un criterio alineado al de las métricas visuales."
                if coherente
                else "lo cual sugiere un criterio más permisivo que el de las métricas en al menos un caso."
            )
        )

    if rejected:
        ids = ", ".join(v["id"] for v in rejected)
        coherente = avg_rej < 0.5
        lines.append(
            f"Las variaciones rechazadas ({ids}) promediaron un MAD de {avg_rej:.2f}, "
            + (
                "coincidiendo con señales de manipulación detectadas automáticamente."
                if coherente
                else "a pesar de un valor relativamente alto, lo que indica un criterio más estricto por parte del usuario."
            )
        )

    lines.append(
        "Este resumen combina la decisión humana con las métricas visuales calculadas "
        "(MAD, FID, LPIPS, ArcFace) y se conserva únicamente durante la sesión activa."
    )
    return "\n\n".join(lines)


def generate_summary(session_json: dict, model_folder: Optional[Path] = None) -> dict:
    """Devuelve {"text": str, "source": "agent"|"rule_based",
    "metrics": dict|None, "approved": bool|None}.

    Intenta primero el AuditAgent real (Gemini, parte 4 del equipo). Si
    no hay GEMINI_API_KEY configurada, o la llamada falla por cualquier
    motivo (sin internet, cuota, formato inesperado), cae al resumen
    basado en reglas — nunca rompe el reporte. `session_json` ya viene
    en el formato exacto que exige el agente (con MAD/FID/LPIPS/ArcFace,
    ver utils/json_manager.py), así que se le pasa tal cual.
    """
    error_detail = None
    try:
        api_key = audit_agent.get_gemini_api_key()
        if not api_key:
            error_detail = "GEMINI_API_KEY no está configurada en st.secrets."
        else:
            agent = audit_agent.AuditAgent()
            result = agent.run(session_json)
            return {
                "text": result["report"],
                "source": "agent",
                "metrics": result["metrics"],
                "approved": result.get("majority_accepted", result.get("approved")),
                "error_detail": None,
            }
    except Exception as e:
        error_detail = f"{type(e).__name__}: {e}"
        print(f"[inference] AuditAgent (Gemini) no disponible, usando resumen local: {error_detail}")

    # el resumen basado en reglas usa nuestras claves internas (MAD ya
    # viene con ese nombre en session_json -- ver json_manager)
    return {
        "text": _rule_based_summary(session_json),
        "source": "rule_based",
        "metrics": None,
        "approved": None,
        "error_detail": error_detail,
    }
