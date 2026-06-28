"""
utils/loader.py
----------------
Carga genérica de modelos desde /modelo_generativo y /modelo_resumen,
soportando las 3 extensiones acordadas: .safetensors, .pkl, .pth

Si la carpeta correspondiente está vacía (todavía no llegó el modelo
real del resto del equipo), devuelve None y el resto del pipeline cae
automáticamente en modo simulado (ver utils/inference.py). Así la
interfaz nunca se rompe por falta de modelo: se puede probar, iterar y
rediseñar de inmediato, y el día que llegue el modelo real basta con
colocar el archivo en la carpeta — no hay que tocar la interfaz.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional, Any

SUPPORTED_EXTENSIONS = (".safetensors", ".pth", ".pkl")


def find_model_file(folder: Path) -> Optional[Path]:
    folder = Path(folder)
    if not folder.exists():
        return None
    for ext in SUPPORTED_EXTENSIONS:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def load_model(folder: Path) -> Optional[Any]:
    """Intenta cargar el primer modelo soportado encontrado en `folder`.
    Devuelve el objeto cargado (state_dict / tensores / objeto pickled)
    o None si no hay modelo disponible o falló la carga.

    IMPORTANTE: esta función carga los *pesos*, no construye la
    arquitectura. Para el modelo generativo (parte 2), si llega como
    archivo en /modelo_generativo en vez de vía backend HTTP, complétalo
    en utils/inference.py -> run_custom_generative_model(). El agente de
    resumen (parte 4) ya no pasa por este loader — ver
    utils/audit_agent.py y la sección 5.1 del README.
    """
    path = find_model_file(folder)
    if path is None:
        return None

    try:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path))

        if path.suffix == ".pth":
            import torch
            return torch.load(str(path), map_location="cpu")

        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)

    except Exception as e:
        print(f"[loader] No se pudo cargar el modelo en {path}: {e}")
        return None

    return None


def model_status(folder: Path) -> str:
    """Texto corto para mostrar en la UI / logs sobre el estado del modelo."""
    path = find_model_file(folder)
    if path is None:
        return "no model found — running in mock mode"
    return f"found {path.name}"
