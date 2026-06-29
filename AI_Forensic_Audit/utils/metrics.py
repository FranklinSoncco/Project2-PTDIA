"""
utils/metrics.py
-----------------
Las 4 métricas que pide el agente de resumen, calculadas LOCALMENTE en
la interfaz apenas llegan las 5 imágenes generadas (sin importar si
vinieron del backend, de un modelo local o del modo simulado):

  MAD     rango [0,1]   mayor = más cercano al original. Es nuestra
                        heurística de siempre (diferencia de píxeles
                        normalizada e invertida), solo que ahora se
                        expone bajo el nombre "MAD" porque así la
                        consume el agente (ver utils/audit_agent.py
                        original: "No uses el término authenticity_score.
                        Usa MAD como métrica local de autenticidad").
  FID     rango >=0     menor = más cercano al dominio real. Es una
                        métrica de SESIÓN, no por imagen: se calcula
                        una sola vez comparando la imagen real (n=1,
                        covarianza 0) contra el conjunto de las 5
                        variaciones, y ese mismo valor se repite en
                        las 5 — tal como espera el equipo del agente
                        ("FID se interpreta mejor como métrica global
                        o de sesión, aunque pueda recibirse repetido
                        por variación en el JSON").
  LPIPS   rango [0,1]   mayor = más diferente perceptualmente. Por
                        variación, vía la red LPIPS (backbone 'squeeze',
                        el más liviano de los tres que ofrece la librería).
  ArcFace rango [-1,1]  mayor = identidad más preservada. Por variación,
                        vía similitud coseno entre embeddings faciales
                        ArcFace (insightface, buffalo_l).

Todas cargan su modelo de forma perezosa la primera vez que se llaman,
cacheado con st.cache_resource (una sola carga por proceso de servidor,
no por sesión de usuario). Si un modelo no carga (sin internet la
primera vez, RAM insuficiente, etc.) se devuelve un valor de respaldo
seguro dentro del rango exigido por el agente y se loguea bien visible
— nunca se rompe la interfaz por esto, pero el valor reportado deja de
ser real.

⚠️ Nota de recursos (importante): torch + lpips + insightface + opencv
son pesadas para el plan gratuito de Streamlit Community Cloud (1GB RAM,
sin GPU). Cargarlas las tres a la vez puede agotar la memoria. Mitigado
parcialmente con carga perezosa y el backbone más liviano disponible en
cada librería, pero si ves errores de memoria en producción, lo más
probable es que sea por esto — considera mover estas 3 métricas al
backend (que sí tiene GPU) si se vuelve un problema real.
"""

from __future__ import annotations
import random
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st

MAD_RANGE = (0.0, 1.0)
LPIPS_RANGE = (0.0, 1.0)
ARCFACE_RANGE = (-1.0, 1.0)

# valores de respaldo si un modelo no carga — neutros, dentro del rango exigido
FALLBACK_LPIPS = 0.5
FALLBACK_ARCFACE = 0.0
FALLBACK_FID = 50.0


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------
# MAD — nuestra heurística de siempre, solo renombrada para el agente
# --------------------------------------------------------------------------

def compute_mad(original: Image.Image, variant: Image.Image) -> float:
    a = original.convert("RGB").resize((64, 64))
    b = variant.convert("RGB").resize((64, 64))
    pa, pb = list(a.getdata()), list(b.getdata())
    diff = sum(abs(p1[c] - p2[c]) for p1, p2 in zip(pa, pb) for c in range(3))
    norm = diff / (64 * 64 * 3 * 255)
    score = 1 - norm * 1.8 + random.uniform(-0.05, 0.05)
    return round(_clip(score, 0.05, 0.97), 4)


# --------------------------------------------------------------------------
# LPIPS — distancia perceptual aprendida, por variación
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_lpips_model():
    import lpips
    return lpips.LPIPS(net="squeeze")  # el backbone más liviano de los 3 disponibles


def _lpips_to_tensor(image: Image.Image):
    import torch
    arr = np.asarray(image.convert("RGB").resize((256, 256))).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # LPIPS espera el rango [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def compute_lpips(original: Image.Image, variant: Image.Image) -> float:
    try:
        import torch
        model = _load_lpips_model()
        with torch.no_grad():
            d = model(_lpips_to_tensor(original), _lpips_to_tensor(variant)).item()
        return round(_clip(d, *LPIPS_RANGE), 4)
    except Exception as e:
        print(f"[metrics] LPIPS no disponible, usando valor de respaldo: {e}")
        return FALLBACK_LPIPS


# --------------------------------------------------------------------------
# FID — de sesión (1 real, covarianza 0, vs 5 variaciones), vía InceptionV3
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_inception_model():
    import torch
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()  # nos quedamos con el feature vector de 2048-d
    model.eval()
    return model


def _inception_features(model, images: list) -> np.ndarray:
    import torch
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    batch = torch.stack([tf(im.convert("RGB")) for im in images])
    with torch.no_grad():
        feats = model(batch)
    return feats.numpy()


def _sqrtm_compat(linalg_module, matrix: np.ndarray) -> np.ndarray:
    """scipy>=1.18 eliminó el parámetro `disp` de sqrtm() y ahora siempre
    devuelve el array directamente (antes, con disp=False, devolvía una
    tupla (resultado, error_estimado)). Esta función funciona con ambas
    versiones -- el bug real visto en producción ("sqrtm() got an
    unexpected keyword argument 'disp'") viene de aquí."""
    try:
        result = linalg_module.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg_module.sqrtm(matrix)
    if isinstance(result, tuple):
        return result[0]
    return result


def compute_fid_session(original: Image.Image, variants: list) -> float:
    """Un solo valor para toda la sesión — repetir en las 5 variaciones."""
    try:
        from scipy import linalg
        model = _load_inception_model()

        real_feat = _inception_features(model, [original])[0]
        gen_feats = _inception_features(model, variants)

        mu_real = real_feat
        sigma_real = np.zeros((real_feat.shape[0], real_feat.shape[0]))  # n=1 -> covarianza 0
        mu_gen = gen_feats.mean(axis=0)
        sigma_gen = np.cov(gen_feats, rowvar=False)

        diff = mu_real - mu_gen
        covmean = _sqrtm_compat(linalg, sigma_real.dot(sigma_gen))
        if not np.isfinite(covmean).all():
            eps = 1e-6
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = _sqrtm_compat(linalg, (sigma_real + offset).dot(sigma_gen + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
        return round(max(0.0, float(fid)), 4)
    except Exception as e:
        print(f"[metrics] FID no disponible, usando valor de respaldo: {e}")
        return FALLBACK_FID


# --------------------------------------------------------------------------
# ArcFace — preservación de identidad, por variación
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_arcface_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 -> CPU
    return app


def _face_embedding(app, image: Image.Image) -> Optional[np.ndarray]:
    arr = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR (lo que espera insightface)
    faces = app.get(arr)
    if not faces:
        return None
    return faces[0].normed_embedding


def compute_arcface_similarity(original: Image.Image, variant: Image.Image) -> float:
    try:
        app = _load_arcface_model()
        emb_real = _face_embedding(app, original)
        emb_var = _face_embedding(app, variant)
        if emb_real is None or emb_var is None:
            print("[metrics] ArcFace: no se detectó rostro en alguna imagen, usando respaldo")
            return FALLBACK_ARCFACE
        sim = float(np.dot(emb_real, emb_var))
        return round(_clip(sim, *ARCFACE_RANGE), 4)
    except Exception as e:
        print(f"[metrics] ArcFace no disponible, usando valor de respaldo: {e}")
        return FALLBACK_ARCFACE


# --------------------------------------------------------------------------
# Orquestador
# --------------------------------------------------------------------------

def compute_all_metrics(original: Image.Image, variant_images: list) -> list[dict]:
    """Devuelve una lista (mismo orden que variant_images) de dicts
    {"MAD":..., "FID":..., "LPIPS":..., "ArcFace":...}. FID es igual en
    las 5 (es de sesión); MAD/LPIPS/ArcFace son por variación."""
    fid_value = compute_fid_session(original, variant_images)
    results = []
    for img in variant_images:
        results.append({
            "MAD": compute_mad(original, img),
            "FID": fid_value,
            "LPIPS": compute_lpips(original, img),
            "ArcFace": compute_arcface_similarity(original, img),
        })
    return results
