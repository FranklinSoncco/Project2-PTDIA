"""
utils/audit_agent.py
---------------------
Agente de resumen (parte 4 del equipo) — reimplementado a partir de
Agente.ipynb (versión con métricas MAD/FID/LPIPS/ArcFace).

Por qué reimplementar en vez de cargar modelo_resumen/agent.pkl
directamente: ese .pkl pesa 41 bytes — es una instancia VACÍA de
AuditAgent, sin estado propio. Toda la lógica real vive en funciones
sueltas del notebook, no en el objeto serializado. Reproducimos la
misma clase y funciones aquí, palabra por palabra según el notebook.

Nota: esta versión del notebook ya NO hardcodea la API key (usa
`google.colab.userdata`) — bien hecho por el equipo. Aquí la leemos
desde st.secrets["GEMINI_API_KEY"], el equivalente para Streamlit.
"""

from __future__ import annotations
import json
import os
import time
from statistics import mean


# --------------------------------------------------------------------------
# Validación (igual que el notebook)
# --------------------------------------------------------------------------

def validate_numeric_metric(variation: dict, field: str, min_value=None, max_value=None) -> None:
    try:
        value = float(variation[field])
    except (TypeError, ValueError):
        raise ValueError(f"{field} debe ser numérico en variación {variation.get('id', 'sin_id')}.")

    if min_value is not None and value < min_value:
        raise ValueError(f"{field} debe ser mayor o igual a {min_value} en variación {variation.get('id', 'sin_id')}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field} debe ser menor o igual a {max_value} en variación {variation.get('id', 'sin_id')}.")


def validate_session_data(data: dict) -> None:
    required_root_fields = ["session_id", "input_image", "reconstruction_b64", "variations"]
    for field in required_root_fields:
        if field not in data:
            raise ValueError(f"Campo raíz faltante: {field}")

    if not isinstance(data["session_id"], str):
        raise ValueError("session_id debe ser string.")
    if not isinstance(data["input_image"], str):
        raise ValueError("input_image debe ser string.")
    if not isinstance(data["reconstruction_b64"], str):
        raise ValueError("reconstruction_b64 debe ser string.")

    variations = data["variations"]
    if not isinstance(variations, list):
        raise ValueError("El campo 'variations' debe ser una lista.")
    if len(variations) != 5:
        raise ValueError("El sistema debe contener exactamente 5 variaciones visuales.")

    required_variation_fields = [
        "id", "label", "description", "image_path", "image_b64",
        "decision", "decision_time_seconds", "MAD", "FID", "LPIPS", "ArcFace",
    ]
    string_variation_fields = ["id", "label", "description", "image_path", "image_b64"]
    valid_decisions = {"accepted", "rejected"}

    for variation in variations:
        if not isinstance(variation, dict):
            raise ValueError("Cada variación debe ser un objeto JSON.")

        for field in required_variation_fields:
            if field not in variation:
                raise ValueError(f"Campo faltante en variación {variation.get('id', 'sin_id')}: {field}")

        for field in string_variation_fields:
            if not isinstance(variation[field], str):
                raise ValueError(f"{field} debe ser string en variación {variation.get('id', 'sin_id')}.")

        if variation["decision"] not in valid_decisions:
            raise ValueError(f"decision debe ser 'accepted' o 'rejected' en variación {variation['id']}.")

        validate_numeric_metric(variation, "decision_time_seconds", min_value=0)
        validate_numeric_metric(variation, "MAD", min_value=0, max_value=1)
        validate_numeric_metric(variation, "FID", min_value=0)
        validate_numeric_metric(variation, "LPIPS", min_value=0, max_value=1)
        validate_numeric_metric(variation, "ArcFace", min_value=-1, max_value=1)


# --------------------------------------------------------------------------
# Métricas (igual que el notebook)
# --------------------------------------------------------------------------

def mean_metric(variations: list, field: str):
    values = [float(v[field]) for v in variations if field in v and v[field] is not None]
    return mean(values) if values else None


def serialize_variation_for_metrics(variation: dict) -> dict:
    return {
        "id": variation["id"],
        "label": variation["label"],
        "description": variation["description"],
        "image_path": variation["image_path"],
        "decision": variation["decision"],
        "decision_time_seconds": float(variation["decision_time_seconds"]),
        "MAD": float(variation["MAD"]),
        "FID": float(variation["FID"]),
        "LPIPS": float(variation["LPIPS"]),
        "ArcFace": float(variation["ArcFace"]),
    }


def compute_metrics(data: dict) -> dict:
    variations = data["variations"]
    total_variations = len(variations)
    if total_variations == 0:
        raise ValueError("No hay variaciones para calcular métricas.")

    accepted = [v for v in variations if v["decision"] == "accepted"]
    rejected = [v for v in variations if v["decision"] == "rejected"]

    decision_times = [float(v["decision_time_seconds"]) for v in variations]
    accepted_decision_times = [float(v["decision_time_seconds"]) for v in accepted]
    rejected_decision_times = [float(v["decision_time_seconds"]) for v in rejected]

    return {
        "session_id": data["session_id"],
        "input_image": data["input_image"],
        "total_variations": total_variations,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": len(accepted) / total_variations,
        "rejection_rate": len(rejected) / total_variations,
        "avg_decision_time_seconds": mean(decision_times),
        "avg_decision_time_accepted_seconds": mean(accepted_decision_times) if accepted_decision_times else None,
        "avg_decision_time_rejected_seconds": mean(rejected_decision_times) if rejected_decision_times else None,
        "avg_MAD": mean_metric(variations, "MAD"),
        "avg_MAD_accepted": mean_metric(accepted, "MAD"),
        "avg_MAD_rejected": mean_metric(rejected, "MAD"),
        "avg_FID": mean_metric(variations, "FID"),
        "avg_FID_accepted": mean_metric(accepted, "FID"),
        "avg_FID_rejected": mean_metric(rejected, "FID"),
        "avg_LPIPS": mean_metric(variations, "LPIPS"),
        "avg_LPIPS_accepted": mean_metric(accepted, "LPIPS"),
        "avg_LPIPS_rejected": mean_metric(rejected, "LPIPS"),
        "avg_ArcFace": mean_metric(variations, "ArcFace"),
        "avg_ArcFace_accepted": mean_metric(accepted, "ArcFace"),
        "avg_ArcFace_rejected": mean_metric(rejected, "ArcFace"),
        "accepted_variations": [serialize_variation_for_metrics(v) for v in accepted],
        "rejected_variations": [serialize_variation_for_metrics(v) for v in rejected],
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


def format_seconds(value) -> str:
    if value is None:
        return "no aplica"
    return f"{float(value):.2f} segundos"


def format_metric(value) -> str:
    if value is None:
        return "no aplica"
    return f"{float(value):.3f}"


def decision_to_spanish(decision: str) -> str:
    if decision == "accepted":
        return "aceptada"
    if decision == "rejected":
        return "rechazada"
    raise ValueError(f"Decisión no válida: {decision}")


def generate_audit_report(data: dict, metrics: dict) -> str:
    accepted = [v for v in data["variations"] if v["decision"] == "accepted"]
    rejected = [v for v in data["variations"] if v["decision"] == "rejected"]

    variaciones_detalle = ""
    for v in data["variations"]:
        decision_es = decision_to_spanish(v["decision"])
        variaciones_detalle += (
            f"- {v['id']} ({v['label']}): {v['description']} "
            f"→ decisión: {decision_es}. "
            f"Tiempo de decisión: {float(v['decision_time_seconds']):.2f} segundos. "
            f"MAD: {float(v['MAD']):.3f}. "
            f"FID: {float(v['FID']):.3f}. "
            f"LPIPS: {float(v['LPIPS']):.3f}. "
            f"ArcFace: {float(v['ArcFace']):.3f}.\n"
        )

    accepted_text = ", ".join(v["id"] + " (" + v["label"] + ")" for v in accepted) if accepted else "ninguna"
    rejected_text = ", ".join(v["id"] + " (" + v["label"] + ")" for v in rejected) if rejected else "ninguna"

    prompt = f"""
Eres un agente de auditoría para un sistema generativo interactivo.

Objetivo:
Generar un reporte textual coherente con las decisiones humanas registradas.

Reglas estrictas:
- Usa solo la información entregada.
- No inventes fechas.
- No inventes métricas.
- No agregues decisiones que no estén en el JSON.
- Mantén alineación exacta entre cada variación y su decisión registrada.
- No uses el término authenticity_score. Usa MAD como métrica local de autenticidad.
- No afirmes haber inspeccionado visualmente las imágenes. Solo recibes metadata textual y métricas calculadas.
- Presenta MAD, FID, LPIPS y ArcFace como métricas de apoyo, no como decisión automática.

Interpretación de métricas:
- MAD: métrica local de autenticidad o cercanía perceptual. Mayor valor indica mayor cercanía local.
- FID: métrica global o de sesión asociada a calidad visual. Menor valor indica mejor cercanía al dominio real.
- LPIPS: diversidad o distancia perceptual. Mayor valor indica mayor diferencia perceptual.
- ArcFace: preservación de identidad. Mayor valor indica mayor similitud de identidad.

Datos generales:
- Session ID: {data["session_id"]}
- Imagen de entrada: {data["input_image"]}
- Total de variaciones evaluadas: {metrics["total_variations"]}

Variaciones aceptadas:
{accepted_text}

Variaciones rechazadas:
{rejected_text}

Detalle por variación:
{variaciones_detalle}

Métricas agregadas:
- Tasa de aceptación humana: {metrics["acceptance_rate"]:.2f}
- Tasa de rechazo humana: {metrics["rejection_rate"]:.2f}
- Tiempo promedio de decisión: {format_seconds(metrics["avg_decision_time_seconds"])}
- Tiempo promedio en aceptadas: {format_seconds(metrics["avg_decision_time_accepted_seconds"])}
- Tiempo promedio en rechazadas: {format_seconds(metrics["avg_decision_time_rejected_seconds"])}
- MAD promedio: {format_metric(metrics["avg_MAD"])}
- FID promedio recibido: {format_metric(metrics["avg_FID"])}
- LPIPS promedio: {format_metric(metrics["avg_LPIPS"])}
- ArcFace promedio: {format_metric(metrics["avg_ArcFace"])}

Genera reporte en español con estas secciones exactas:

# Reporte de auditoría visual

## Información general
## Resumen de decisiones
## Análisis por variación
## Métricas de evaluación
## Conclusión
## Limitaciones

En "Análisis por variación", menciona explícitamente:
id, label, descripción, decisión, tiempo de decisión, MAD, FID, LPIPS y ArcFace.

En "Métricas de evaluación", explica brevemente qué significa cada métrica y resume los promedios.

En "Conclusión", resume el patrón general de aceptación/rechazo usando decisiones humanas, labels, descripciones y métricas disponibles.

En "Limitaciones", aclara que las métricas son evidencia complementaria y que la decisión final proviene del usuario.
"""

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada (st.secrets / variable de entorno)")

    import google.generativeai as genai  # noqa: PLC0415 (import perezoso a propósito)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0.0})
    response = model.generate_content(prompt)
    return response.text


# --------------------------------------------------------------------------
# Evaluación de coherencia del reporte (igual que el notebook)
# --------------------------------------------------------------------------

def metric_value_mentioned(report_lower: str, value: float) -> bool:
    value = float(value)
    possible_formats = [f"{value:.3f}", f"{value:.2f}", str(value)]
    return any(metric_text in report_lower for metric_text in possible_formats)


def evaluate_report_alignment(data: dict, report: str) -> dict:
    variations = data["variations"]
    report_lower = report.lower()

    mentioned_count = 0
    correct_decision_count = 0
    label_mentioned_count = 0
    description_mentioned_count = 0
    decision_time_mentioned_count = 0
    mad_mentioned_count = 0
    fid_mentioned_count = 0
    lpips_mentioned_count = 0
    arcface_mentioned_count = 0

    accepted_keywords = ["aceptada", "aceptado", "accepted"]
    rejected_keywords = ["rechazada", "rechazado", "rejected"]

    for variation in variations:
        variation_id = variation["id"].lower()
        label = variation["label"].lower()
        description = variation["description"].lower()
        decision_time = float(variation["decision_time_seconds"])

        position = report_lower.find(variation_id)
        if position != -1:
            mentioned_count += 1
            local_window = report_lower[max(0, position - 160): position + 500]
            if variation["decision"] == "accepted" and any(k in local_window for k in accepted_keywords):
                correct_decision_count += 1
            if variation["decision"] == "rejected" and any(k in local_window for k in rejected_keywords):
                correct_decision_count += 1

        if label in report_lower:
            label_mentioned_count += 1
        if description in report_lower:
            description_mentioned_count += 1
        if metric_value_mentioned(report_lower, decision_time):
            decision_time_mentioned_count += 1
        if "mad" in report_lower and metric_value_mentioned(report_lower, variation["MAD"]):
            mad_mentioned_count += 1
        if "fid" in report_lower and metric_value_mentioned(report_lower, variation["FID"]):
            fid_mentioned_count += 1
        if "lpips" in report_lower and metric_value_mentioned(report_lower, variation["LPIPS"]):
            lpips_mentioned_count += 1
        if "arcface" in report_lower and metric_value_mentioned(report_lower, variation["ArcFace"]):
            arcface_mentioned_count += 1

    total_variations = len(variations)
    if total_variations == 0:
        raise ValueError("No hay variaciones para evaluar alineación.")

    visual_metric_coverage = (
        mad_mentioned_count + fid_mentioned_count + lpips_mentioned_count + arcface_mentioned_count
    ) / (total_variations * 4)

    return {
        "explanation_coverage": mentioned_count / total_variations,
        "decision_summary_coherence": correct_decision_count / total_variations,
        "label_coverage": label_mentioned_count / total_variations,
        "description_coverage": description_mentioned_count / total_variations,
        "decision_time_coverage": decision_time_mentioned_count / total_variations,
        "MAD_coverage": mad_mentioned_count / total_variations,
        "FID_coverage": fid_mentioned_count / total_variations,
        "LPIPS_coverage": lpips_mentioned_count / total_variations,
        "ArcFace_coverage": arcface_mentioned_count / total_variations,
        "visual_metric_coverage": visual_metric_coverage,
    }


# --------------------------------------------------------------------------
# Ejecución completa
# --------------------------------------------------------------------------

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

    def run_from_file(self, json_path) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        return self.run(session_data)

    def run(self, session_data: dict) -> dict:
        metrics, report = run_language_agent(session_data)
        return {
            "report": report,
            "metrics": metrics,
            "majority_accepted": metrics["accepted_count"] > metrics["rejected_count"],
            "accepted_count": metrics["accepted_count"],
            "rejected_count": metrics["rejected_count"],
            "total_variations": metrics["total_variations"],
            "avg_MAD": metrics["avg_MAD"],
            "avg_FID": metrics["avg_FID"],
            "avg_LPIPS": metrics["avg_LPIPS"],
            "avg_ArcFace": metrics["avg_ArcFace"],
            "decision_summary_coherence": metrics["decision_summary_coherence"],
            "visual_metric_coverage": metrics["visual_metric_coverage"],
        }
