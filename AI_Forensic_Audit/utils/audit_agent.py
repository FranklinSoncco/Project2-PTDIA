"""
utils/audit_agent.py
---------------------
Agente de resumen (parte 4 del equipo) — reimplementado a partir de
proyecto2_agente.ipynb.

Por qué reimplementar en vez de cargar modelo_resumen/agent.pkl
directamente: ese .pkl pesa 41 bytes — es una instancia VACÍA de
AuditAgent, sin ningún estado/peso propio. Toda la lógica real vive en
funciones sueltas del notebook (validate_session_data, compute_metrics,
generate_audit_report, evaluate_report_alignment), no en el objeto
serializado. Además, el pickle quedó referenciado a `__main__.AuditAgent`,
lo cual es frágil de deserializar fuera del notebook original. Por eso
reproducimos la misma clase y funciones aquí palabra por palabra — el
resultado es exactamente equivalente, sin la fragilidad del pickle.
El archivo .pkl se conserva en modelo_resumen/ igual, como referencia.

⚠️ SEGURIDAD: el notebook original tenía la API key de Gemini escrita
en texto plano en el código. NO se copió aquí a propósito — avísale al
equipo que la rote (una key hardcodeada que llega a un repo compartido
queda expuesta para siempre en el historial de git, aunque se borre
después). Esta versión la lee desde st.secrets["GEMINI_API_KEY"].

⚠️ SDK DEPRECADO: `google.generativeai` (el paquete que usa el
notebook) ya no recibe actualizaciones — Google migró todo a un SDK
unificado nuevo, `google-genai` (`from google import genai`). Se deja
el código TAL CUAL lo probó el equipo (es lo único que confirmamos que
funciona de verdad, porque yo no tengo acceso a la API de Gemini desde
mi entorno para probar una migración). Si en el futuro deja de
funcionar, la migración es mecánica — ver la guía oficial:
https://ai.google.dev/gemini-api/docs/migrate
"""

from __future__ import annotations
import json
import os
import time
from statistics import mean


# --------------------------------------------------------------------------
# Validación (igual que el notebook)
# --------------------------------------------------------------------------

def validate_session_data(data: dict) -> None:
    required_root_fields = ["session_id", "input_image", "reconstruction_b64", "variations"]
    for field in required_root_fields:
        if field not in data:
            raise ValueError(f"Campo raíz faltante: {field}")

    variations = data["variations"]
    if not isinstance(variations, list):
        raise ValueError("El campo 'variations' debe ser una lista.")
    if len(variations) != 5:
        raise ValueError("El sistema debe contener exactamente 5 variaciones visuales.")

    required_variation_fields = [
        "id", "label", "description", "image_path", "image_b64",
        "decision", "decision_time_seconds",
    ]
    valid_decisions = {"accepted", "rejected"}

    for variation in variations:
        for field in required_variation_fields:
            if field not in variation:
                raise ValueError(f"Campo faltante en variación: {field}")
        if variation["decision"] not in valid_decisions:
            raise ValueError("decision debe ser 'accepted' o 'rejected'.")
        decision_time = float(variation["decision_time_seconds"])
        if decision_time < 0:
            raise ValueError("decision_time_seconds debe ser mayor o igual a 0.")


# --------------------------------------------------------------------------
# Métricas (igual que el notebook)
# --------------------------------------------------------------------------

def compute_metrics(data: dict) -> dict:
    variations = data["variations"]
    accepted = [v for v in variations if v["decision"] == "accepted"]
    rejected = [v for v in variations if v["decision"] == "rejected"]

    decision_times = [float(v["decision_time_seconds"]) for v in variations]
    accepted_decision_times = [float(v["decision_time_seconds"]) for v in accepted]
    rejected_decision_times = [float(v["decision_time_seconds"]) for v in rejected]

    return {
        "session_id": data["session_id"],
        "input_image": data["input_image"],
        "total_variations": len(variations),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": len(accepted) / len(variations),
        "rejection_rate": len(rejected) / len(variations),
        "avg_decision_time_seconds": mean(decision_times) if decision_times else None,
        "avg_decision_time_accepted_seconds": mean(accepted_decision_times) if accepted_decision_times else None,
        "avg_decision_time_rejected_seconds": mean(rejected_decision_times) if rejected_decision_times else None,
        "accepted_variations": [
            {
                "id": v["id"], "label": v["label"], "description": v["description"],
                "image_path": v["image_path"], "decision": v["decision"],
                "decision_time_seconds": float(v["decision_time_seconds"]),
            }
            for v in accepted
        ],
        "rejected_variations": [
            {
                "id": v["id"], "label": v["label"], "description": v["description"],
                "image_path": v["image_path"], "decision": v["decision"],
                "decision_time_seconds": float(v["decision_time_seconds"]),
            }
            for v in rejected
        ],
    }


# --------------------------------------------------------------------------
# Agente basado en lenguaje (Gemini) — igual que el notebook, salvo la API key
# --------------------------------------------------------------------------

def get_gemini_api_key() -> str:
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def generate_audit_report(data: dict, metrics: dict) -> str:
    accepted = [v for v in data["variations"] if v["decision"] == "accepted"]
    rejected = [v for v in data["variations"] if v["decision"] == "rejected"]

    variaciones_detalle = ""
    for v in data["variations"]:
        variaciones_detalle += (
            f"- {v['id']} ({v['label']}): {v['description']} "
            f"→ {v['decision']} ({v['decision_time_seconds']}s)\n"
        )

    prompt = f"""
    Eres un agente auditor.
    Genera un reporte utilizando únicamente la información proporcionada.

    Datos de la sesión:
    - Variaciones ACEPTADAS: {", ".join(v["id"] + " (" + v["label"] + ")" for v in accepted) if accepted else "ninguna"}
    - Variaciones RECHAZADAS: {", ".join(v["id"] + " (" + v["label"] + ")" for v in rejected) if rejected else "ninguna"}

    Detalle de cada variación:
    {variaciones_detalle}

    Estadísticas:
    - Tasa de aceptación: {metrics["acceptance_rate"]:.0%}
    - Tiempo promedio de decisión: {metrics["avg_decision_time_seconds"]:.2f} segundos
    - Tiempo promedio en aceptadas: {metrics["avg_decision_time_accepted_seconds"]:.2f} segundos
    - Tiempo promedio en rechazadas: {metrics["avg_decision_time_rejected_seconds"]:.2f} segundos

    El reporte debe incluir:
    ## Información general
    ## Variaciones aceptadas
    ## Variaciones rechazadas
    ## Descripción de cada variación
    ## Métricas
    ## Conclusión
    ## Limitaciones

    No infieras gustos, emociones o motivaciones del usuario.
    No inventes información que no aparezca en los datos.
    """

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada (st.secrets / variable de entorno)")

    import google.generativeai as genai  # noqa: PLC0415 (import perezoso a propósito)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


def evaluate_report_alignment(data: dict, report: str) -> dict:
    variations = data["variations"]
    report_lower = report.lower()
    mentioned_count = 0
    correct_decision_count = 0
    label_mentioned_count = 0
    description_mentioned_count = 0
    image_path_mentioned_count = 0
    decision_time_mentioned_count = 0

    for variation in variations:
        variation_id = variation["id"].lower()
        label = variation["label"].lower()
        description = variation["description"].lower()
        image_path = variation["image_path"].lower()
        decision_time = float(variation["decision_time_seconds"])

        if variation_id in report_lower:
            mentioned_count += 1
        if label in report_lower:
            label_mentioned_count += 1
        if description in report_lower:
            description_mentioned_count += 1
        if image_path in report_lower:
            image_path_mentioned_count += 1
        if f"{decision_time:.2f}" in report_lower:
            decision_time_mentioned_count += 1

        accepted_keywords = ["aceptada", "aceptado", "accepted"]
        rejected_keywords = ["rechazada", "rechazado", "rejected"]

        if variation["decision"] == "accepted":
            if variation_id in report_lower and any(kw in report_lower for kw in accepted_keywords):
                correct_decision_count += 1
        else:
            if variation_id in report_lower and any(kw in report_lower for kw in rejected_keywords):
                correct_decision_count += 1

    total_variations = len(variations)
    return {
        "explanation_coverage": mentioned_count / total_variations,
        "decision_summary_coherence": correct_decision_count / total_variations,
        "label_coverage": label_mentioned_count / total_variations,
        "description_coverage": description_mentioned_count / total_variations,
        "image_path_coverage": image_path_mentioned_count / total_variations,
        "decision_time_coverage": decision_time_mentioned_count / total_variations,
    }


def run_language_agent(data: dict) -> tuple:
    start_time = time.time()
    validate_session_data(data)
    metrics = compute_metrics(data)
    report = generate_audit_report(data, metrics)
    alignment_metrics = evaluate_report_alignment(data, report)
    metrics.update(alignment_metrics)
    metrics["agent_latency_seconds"] = round(time.time() - start_time, 4)
    return metrics, report


class AuditAgent:
    """Reimplementación fiel del AuditAgent del notebook del equipo."""

    def run_from_file(self, json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        return self.run(session_data)

    def run(self, session_data: dict) -> dict:
        metrics, report = run_language_agent(session_data)
        return {
            "report": report,
            "metrics": metrics,
            "approved": metrics["accepted_count"] > metrics["rejected_count"],
        }
