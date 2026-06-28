"""
utils/api_client.py
--------------------
Cliente HTTP para el backend externo del equipo (URL permanente,
confirmada por el equipo de modelo generativo). Streamlit Cloud NO
corre el modelo pesado — solo llama a esta API y usa lo que devuelva.
Si el backend no responde, la interfaz cae automáticamente al modo
simulado en vez de romperse.

Contrato CONFIRMADO con el equipo (swagger /docs):

  POST /generate
    request:  multipart/form-data, campo "file" (la imagen)
    response:
      {
        "session_id": "ab12cd34",
        "input_image": "/files/ab12cd34/input.png",
        "reconstruction_b64": "<png base64>",
        "variations": [
          {"id": "V1", "label": "Mayor edad", "description": "...",
           "image_path": "/files/ab12cd34/var_1.png", "image_b64": "<png base64>"},
          ... (V2..V5)
        ]
      }

    Nota sobre el score: el contrato que compartiste no incluye un
    campo de métrica, pero se intenta leer bajo varios nombres comunes
    (ver SCORE_KEYS) por si el equipo lo agrega o lo llama distinto a
    lo esperado. Si de verdad no viene en la respuesta, se usa un
    cálculo local SOLO como red de seguridad para no romper la
    interfaz — y se loguea bien visible para que no pase desapercibido,
    porque el score real debe venir del backend.

    El "session_id" que devuelven ES DE ELLOS (no el nuestro) y hay que
    reusarlo al mandar /feedback para que puedan correlacionar la
    auditoría. "label"/"description" por variación se guardan en
    memoria por si se quieren mostrar en la interfaz más adelante, pero
    no se escriben en nuestro session.json (ese formato se mantiene
    como lo planeamos para el modelo de resumen).

  POST /feedback   (para no dejar un hueco del lado de ellos)
    request:  application/json
              {"session_id": "<el de ellos>", "decisions": [
                  {"id": "V1", "accepted": true, "authenticity_score": 0.86}, ...
              ]}
    "Best effort": si falla, se loguea pero no rompe el reporte (que ya
    se construyó localmente con nuestro propio formato/resumen).
"""

from __future__ import annotations
import base64
import io
from typing import Optional

import requests
from PIL import Image

DEFAULT_TIMEOUT = 90  # segundos — generación de imágenes puede tardar
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}  # evita la página de aviso de ngrok

# nombres de campo candidatos para la métrica, en orden de prioridad
SCORE_KEYS = ("authenticity_score", "score", "metric", "confidence", "auth_score")


def backend_is_configured(backend_url: Optional[str]) -> bool:
    return bool(backend_url and backend_url.strip())


def get_backend_url() -> str:
    """Lee BACKEND_URL desde .streamlit/secrets.toml. Si no está
    configurado, todo el pipeline cae al modo simulado / sin feedback
    remoto."""
    try:
        import streamlit as st
        return st.secrets.get("BACKEND_URL", "")
    except Exception:
        return ""


def _decode_b64_image(b64_str: str) -> Image.Image:
    if b64_str.strip().startswith("data:") and "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _extract_score(v: dict) -> Optional[float]:
    """Busca la métrica de autenticidad bajo varios nombres posibles.
    Devuelve None si el backend de verdad no la incluyó."""
    for key in SCORE_KEYS:
        if key in v and v[key] is not None:
            try:
                return float(v[key])
            except (TypeError, ValueError):
                continue
    return None


def call_generative_backend(
    backend_url: str,
    image: Image.Image,
    n: int,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """Envía la imagen a POST {BACKEND_URL}/generate. Devuelve:
        {"backend_session_id": str|None,
         "variations": [{"image": PIL.Image, "score": float|None, "meta": {...}}, ...]}
    o None si algo falla — generate_variations() cae al modo simulado
    cuando esto devuelve None, así que nunca tumba la interfaz.
    """
    if not backend_url:
        return None

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{backend_url.rstrip('/')}/generate",
            files={"file": ("input.png", buf, "image/png")},
            headers=NGROK_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"[api_client] Backend /generate no disponible o respuesta inválida: {e}")
        return None

    variations = []
    missing_score_ids = []
    for v in payload.get("variations", [])[:n]:
        try:
            img = _decode_b64_image(v["image_b64"])
        except Exception as e:
            print(f"[api_client] No se pudo leer una variación de la respuesta: {e}")
            continue
        score = _extract_score(v)
        if score is None:
            missing_score_ids.append(v.get("id", "?"))
        variations.append({
            "image": img,
            "score": score,
            "meta": {"id": v.get("id"), "label": v.get("label"), "description": v.get("description")},
        })

    if missing_score_ids:
        print(
            f"[api_client] ⚠️  El backend NO mandó métrica para: {missing_score_ids} "
            f"(se probaron las claves {SCORE_KEYS}). Usando cálculo local de respaldo "
            f"SOLO para esas — confirma con el equipo de dónde debe salir el score real."
        )

    if not variations:
        return None

    return {
        "backend_session_id": payload.get("session_id"),
        "variations": variations,
    }


def send_feedback(
    backend_url: str,
    session_id: str,
    decisions: list[dict],
    timeout: int = 30,
) -> bool:
    """POST {BACKEND_URL}/feedback. "Best effort": si falla, devuelve
    False y solo se loguea — el reporte local no depende de esto."""
    if not backend_url:
        return False

    try:
        resp = requests.post(
            f"{backend_url.rstrip('/')}/feedback",
            json={"session_id": session_id, "decisions": decisions},
            headers=NGROK_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[api_client] No se pudo enviar feedback al backend: {e}")
        return False
