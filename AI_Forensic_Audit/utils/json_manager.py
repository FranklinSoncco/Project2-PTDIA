"""
utils/json_manager.py
----------------------
Construye y persiste el JSON que consume el agente de resumen (parte 4
del equipo). Formato confirmado por el equipo de NLP — coincide con la
salida de /generate (session_id, input_image, reconstruction_b64,
variations con id/label/description/image_path/image_b64) más
decision/decision_time_seconds que agrega nuestra interfaz. Se incluye
además authenticity_score (no lo exige el agente, pero seguimos
calculándolo localmente y es útil para nuestro propio reporte/export).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any


def build_session_json(
    session_id: str,
    input_image_rel: str,
    reconstruction_b64: str,
    variations: List[Dict[str, Any]],
) -> dict:
    return {
        "session_id": session_id,
        "input_image": input_image_rel,
        "reconstruction_b64": reconstruction_b64,
        "variations": [
            {
                "id": v["id"],
                "label": v.get("label") or "Variación",
                "description": v.get("description") or "",
                "image_path": v["image_path"],
                "image_b64": v["image_b64"],
                "decision": v.get("decision"),
                "decision_time_seconds": v.get("decision_time_seconds"),
                "authenticity_score": round(float(v["authenticity_score"]), 4),
            }
            for v in variations
        ],
    }


def build_export_json(session_id: str, variations: List[Dict[str, Any]]) -> dict:
    """Versión liviana para el botón 'Export JSON' del usuario — sin
    los base64 (harían el archivo enorme e ilegible para una persona)."""
    return {
        "session_id": session_id,
        "variations": [
            {
                "id": v["id"],
                "label": v.get("label") or "Variación",
                "description": v.get("description") or "",
                "image_path": v["image_path"],
                "authenticity_score": round(float(v["authenticity_score"]), 4),
                "decision": v.get("decision"),
                "decision_time_seconds": v.get("decision_time_seconds"),
            }
            for v in variations
        ],
    }


def save_json(data: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
