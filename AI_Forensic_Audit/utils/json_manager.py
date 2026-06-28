"""
utils/json_manager.py
----------------------
Construye y persiste el JSON que consume el modelo/agente de resumen
(parte 4 del equipo). Formato actualizado por el equipo (nombres en
inglés + métrica de tiempo de decisión):

{
  "session_id": "audit_001",
  "input_image": "inputs/rostro_real_001.jpg",
  "variations": [
    {
      "id": "V1",
      "image_path": "outputs/var_1.png",
      "authenticity_score": 0.86,
      "decision": "accepted",
      "decision_time_seconds": 5.8
    },
    ...
  ]
}
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any


def build_session_json(session_id: str, input_image_rel: str, variations: List[Dict[str, Any]]) -> dict:
    return {
        "session_id": session_id,
        "input_image": input_image_rel,
        "variations": [
            {
                "id": v["id"],
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
