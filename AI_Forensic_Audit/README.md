# DEVERITAS

**Asistente forense de verificación visual** — interfaz de interacción
humano-IA para el Proyecto #2 (Planificación y Toma de Decisiones en
IA, UTEC). Parte 3 del equipo: la interfaz.

> El usuario carga un rostro, el sistema genera 5 variaciones
> controladas con un score de autenticidad cada una, el usuario aprueba
> o rechaza cada variación, y un agente sintetiza un reporte de
> auditoría que combina la decisión humana con la señal del modelo.

---

## 1. Cómo correrlo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Con eso ya queda funcionando **de punta a punta en modo simulado**: no
hace falta ningún modelo entrenado para probar, iterar o rediseñar la
interfaz. Las "variaciones" se generan con transformaciones de imagen
(PIL) y el score de autenticidad con una heurística simple — suficiente
para validar todo el flujo mientras el resto del equipo entrena los
modelos reales.

La interfaz no muestra ningún aviso de "modo simulado" — para el
usuario final el flujo se ve igual ya sea con datos simulados o con los
modelos reales conectados. Si quieres confirmar qué está usando en un
momento dado, revisa `utils/loader.model_status()` o los logs de la
terminal donde corre `streamlit run app.py`.

## 2. Cómo conectar los modelos reales del equipo

Cuando lleguen los modelos de las partes 2 (generativo) y 4 (resumen):

1. Coloca el archivo de pesos (`.safetensors`, `.pth` o `.pkl`) dentro
   de `modelo_generativo/` o `modelo_resumen/`. No hace falta tocar la
   interfaz ni el resto del pipeline.
2. Completa la función correspondiente en `utils/inference.py`:
   - `run_custom_generative_model(model, image, n)` → debe devolver una
     lista de `n` tuplas `(PIL.Image, score_float)`.
   - `run_custom_summary_model(model, session_json)` → debe devolver el
     texto del resumen (string).
3. Listo — `utils/loader.py` ya se encarga de detectar el archivo y
   cargarlo; si no encuentra nada, sigue en modo simulado sin romper
   nada.

### Descargar un modelo abierto para probar con algo real

```bash
pip install -r requirements-models.txt
python download_models.py
```

Esto descarga un modelo de difusión abierto (variaciones de imagen)
para tener algo "real" con qué probar mientras el equipo entrena el
modelo definitivo del problema forense. **Nota:** este script no se
pudo ejecutar dentro del entorno donde se generó este proyecto (sin
acceso a Hugging Face), así que está revisado por sintaxis pero no
probado con una descarga real — pruébalo en tu máquina o en Colab.

### Opción B: backend externo (lo que estamos usando ahora)

El modelo generativo es pesado para correr en Streamlit Cloud (CPU,
RAM limitada), así que el equipo lo expone como una API HTTP aparte y
la interfaz solo le hace una llamada. La URL es **permanente**
(confirmado por el equipo de modelo generativo), así que no debería
cambiar entre despliegues.

1. Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml`
   (local) **o**, si ya está en Streamlit Cloud, pega la misma línea en
   tu app → *Settings → Secrets*:
   ```toml
   BACKEND_URL = "https://uptake-pregnant-obstruct.ngrok-free.dev"
   ```
2. Listo — `utils/api_client.py` le hace `POST {BACKEND_URL}/generate`
   con la imagen y decodifica las 5 variaciones (`image_b64`) que
   devuelve. Si el backend no responde, cae automáticamente al modo
   simulado.

**Contrato confirmado** (`/generate`):
```json
{
  "session_id": "ab12cd34",
  "input_image": "/files/ab12cd34/input.png",
  "reconstruction_b64": "<png base64>",
  "variations": [
    {"id": "V1", "label": "Mayor edad", "description": "...",
     "image_path": "/files/ab12cd34/var_1.png", "image_b64": "<png base64>"}
  ]
}
```
- No incluye `authenticity_score` — se calcula **localmente**
  (`utils/inference.py::_diff_score`), igual que en modo simulado.
- El `session_id` que devuelven es **de ellos**, no el nuestro — se
  guarda en `st.session_state.backend_session_id` y se reutiliza al
  mandar `/feedback`, para que puedan correlacionar la auditoría de su
  lado.
- `label`/`description` por variación se guardan en memoria por si
  más adelante quieres mostrarlos en la tarjeta (por ejemplo, "Mayor
  edad" debajo del score) — hoy no se muestran en la UI ni se
  escriben en `session.json` para no tocar ese formato.
- `reconstruction_b64` no se usa todavía.

**`/feedback`** — para no dejar un hueco del lado de ellos, al
terminar las 5 decisiones se manda automáticamente:
```json
{"session_id": "<el de ellos>", "decisions": [
  {"id": "V1", "accepted": true, "authenticity_score": 0.86}
]}
```
Esto es independiente de **nuestro** `session.json` / resumen local,
que se mantiene con el formato que ya tenían planeado (`decision`:
`"accepted"`/`"rejected"`, `decision_time_seconds`, etc.) — ver sección 5.
Si `/feedback` falla, solo se loguea; el reporte no depende de eso.

## 3. Estructura del proyecto

```
AI_Forensic_Audit/
├── app.py                     # orquesta el state machine de pantallas
├── requirements.txt           # streamlit + pillow + numpy + requests + google-generativeai
├── requirements-models.txt    # torch/diffusers/transformers (modelo generativo local, opcional)
├── download_models.py         # descarga un modelo abierto de prueba (alternativa al backend)
├── test_backend_connection.py # prueba /generate directo contra el backend real + guarda imágenes
│
├── assets/
│   ├── logo.svg             # logo (escudo + ojo + red neuronal + check)
│   ├── favicon.png
│   └── css/styles.css       # design system compartido con el prototipo HTML
│
├── components/              # una función de render por pantalla
│   ├── navbar.py
│   ├── landing.py
│   ├── uploader.py
│   ├── generator.py
│   ├── cards.py
│   └── report.py
│
├── utils/
│   ├── session.py            # UUID + carpetas por sesión (multi-usuario)
│   ├── cleanup.py            # destruye datos al terminar / sesiones abandonadas
│   ├── json_manager.py       # construye/lee el JSON de auditoría (formato del agente)
│   ├── loader.py             # carga .safetensors/.pth/.pkl con fallback a mock
│   ├── inference.py          # generación + resumen (mock, backend, hooks reales)
│   ├── metrics.py            # MAD, FID, LPIPS, ArcFace — calculadas siempre localmente
│   ├── api_client.py         # cliente HTTP del backend externo (/generate, /feedback)
│   ├── audit_agent.py        # AuditAgent real (Gemini) — parte 4, reimplementado del notebook
│   ├── svg.py                # logo, íconos propios, dial de score
│   └── styling.py            # inyección del CSS global
│
├── modelo_generativo/        # alternativa local al backend (parte 2), opcional
├── modelo_resumen/
│   └── agent.pkl              # el que entregó el equipo — solo de referencia, ver sección 5.1
└── sessions/                 # se crea en runtime, una carpeta UUID por usuario
```

## 4. Arquitectura de sesiones (multi-usuario sin cruces)

Cada usuario conectado corre en su propio `st.session_state` (Streamlit
ya aísla eso). Lo que se comparte es el disco, así que cada sesión
recibe una carpeta con UUID:

```
sessions/audit_<uuid>/
    input/   -> imagen original
    output/  -> las 5 variaciones generadas
    json/    -> session.json (scores + decisiones)
    logs/
```

- Al terminar una auditoría (botón **New Audit**) la carpeta de esa
  sesión se destruye por completo (`shutil.rmtree`).
- Como red de seguridad, `utils/cleanup.purge_stale_sessions()` borra
  cualquier carpeta abandonada (usuario cerró la pestaña sin terminar)
  con más de 30 minutos de antigüedad. Se llama de forma oportunista en
  cada arranque de `app.py`.

Esto permite que PEPE1 y PEPE2 usen la app al mismo tiempo sin que sus
imágenes, variaciones o decisiones se mezclen.

## 5. Formato del JSON de auditoría

**Actualizado de nuevo** por el equipo de NLP (parte 4) — ahora exige 4
métricas visuales por variación además de todo lo anterior:

```json
{
  "session_id": "audit_a1b2c3d4e5",
  "input_image": "inputs/source.jpg",
  "reconstruction_b64": "<png base64>",
  "variations": [
    {
      "id": "V1",
      "label": "Mayor edad",
      "description": "Variación con rasgos de mayor edad.",
      "image_path": "outputs/var_1.png",
      "image_b64": "<png base64>",
      "decision": "accepted",
      "decision_time_seconds": 5.8,
      "MAD": 0.86,
      "FID": 12.4,
      "LPIPS": 0.18,
      "ArcFace": 0.93
    }
  ]
}
```

- `decision` es `"accepted"` o `"rejected"`. `decision_time_seconds` se
  mide desde que la variación aparece en pantalla hasta que el usuario
  decide (`components/cards.py`).
- **Sobre el nombre `MAD`**: el agente lo exige con ese nombre exacto
  (su `validate_session_data` lo busca como `"MAD"`, no como
  `"authenticity_score"` — su propio *prompt* incluso le dice al LLM
  *"No uses el término authenticity_score. Usa MAD"*). Internamente
  seguimos llamándola `authenticity_score` en nuestra UI ("AUTHENTICITY
  SCORE"); solo se renombra a `"MAD"` al construir este JSON
  (`utils/json_manager.py`). Es la misma heurística de siempre.
- `FID` es una métrica **de sesión**, no por imagen — se calcula una
  sola vez (imagen real vs. el conjunto de las 5 variaciones) y ese
  mismo valor se repite en las 5, tal como espera el agente.
- `LPIPS` y `ArcFace` sí son por variación.
- Rangos exigidos por `validate_session_data` (si no se respetan, el
  agente *rechaza* el JSON): `MAD∈[0,1]`, `FID≥0`, `LPIPS∈[0,1]`,
  `ArcFace∈[-1,1]`. `utils/metrics.py` ya garantiza estos rangos
  siempre, incluso en modo de respaldo.
- Este JSON (con los base64 completos) es el que se guarda en
  `sessions/<id>/json/session.json` y se le pasa al agente de resumen.
  El botón **Export JSON** de la interfaz descarga una versión más
  liviana, sin los base64, pensada para que una persona la pueda abrir
  y leer.

## 5.1. Las 4 métricas (MAD, FID, LPIPS, ArcFace) — `utils/metrics.py`

Se calculan **siempre localmente en la interfaz**, apenas llegan las 5
imágenes (sin importar si vinieron del backend, de un modelo local o
del modo simulado) — así se decidió, en vez de depender de que el
backend mande su propia métrica.

| Métrica | Librería | Qué mide | Por variación o por sesión |
|---|---|---|---|
| MAD | propia (PIL/numpy) | cercanía de píxeles con el original | por variación |
| FID | `torchvision` (InceptionV3) + `scipy` | distancia de distribución vs. dominio real | **de sesión** (repetida en las 5) |
| LPIPS | `lpips` (backbone `squeeze`) | distancia perceptual aprendida | por variación |
| ArcFace | `insightface` (`buffalo_l`) | preservación de identidad (similitud coseno) | por variación |

Las 4 se muestran en la tarjeta de evaluación, verticales al costado
de la imagen.

⚠️ **Riesgo real de despliegue — leer antes de subir esto a producción**:
`torch` + `torchvision` + `lpips` + `insightface` + `opencv` son
pesadas. Streamlit Community Cloud (plan gratis) da 1 GB de RAM, sin
GPU, y un *build* con límite de tiempo. Cargar los 3 modelos (Inception,
LPIPS, ArcFace) a la vez puede:
- hacer que el *build* tarde varios minutos o falle por espacio,
- quedarse sin memoria en tiempo de ejecución (el proceso se reinicia solo,
  sin aviso claro al usuario).

No pude probar la descarga real de ninguno de los 3 modelos desde mi
entorno (la red del sandbox bloquea `download.pytorch.org` y los
*release assets* de GitHub) — sí verifiqué la lógica matemática de
cada métrica con datos sintéticos, y que **si un modelo falla en
cargar, la interfaz cae a un valor de respaldo seguro dentro del rango
exigido en vez de romperse** (ver `utils/metrics.py`). Pero la
*calidad real* de las 3 métricas pesadas solo se puede confirmar
desplegando de verdad. Si ves errores de memoria en producción:
- Lo más probable es que sea por esto, no por un bug.
- Mitigación rápida: probar primero solo con MAD activo (comentar las
  otras 3 llamadas en `compute_all_metrics`) para aislar cuál pesa más.
- Mitigación de fondo: mover FID/LPIPS/ArcFace al backend (que sí tiene
  GPU) en vez de la interfaz, si esto se vuelve bloqueante.

## 5.2. Agente de resumen real (Gemini)

El equipo de NLP entregó `agent.pkl` + un notebook de prueba
(`Agente.ipynb`). Ese `.pkl` pesa 41 bytes — es una instancia
**vacía** de una clase `AuditAgent` (sin estado propio); la lógica real
vive en funciones del notebook que llaman a Gemini. Por eso
`utils/audit_agent.py` **reimplementa esas funciones directamente**
(palabra por palabra según el notebook, incluida la validación estricta
de rangos de MAD/FID/LPIPS/ArcFace) en vez de deserializar el `.pkl` —
es exactamente equivalente, pero sin la fragilidad de depender de
`pickle.load()` resolviendo una clase que vivía en `__main__` del
notebook original. El `.pkl` se conserva en `modelo_resumen/` solo como
referencia/documentación.

**Para activarlo:** agrega a tus secrets:
```toml
GEMINI_API_KEY = "tu-api-key-real"
```
Si no está configurada, o la llamada a Gemini falla por cualquier
motivo, la interfaz cae automáticamente al resumen basado en reglas —
nunca rompe el reporte. La interfaz muestra cuál se usó (`resumen vía
agente (Gemini)` / `resumen basado en reglas`), igual que con la fuente
de las imágenes.

✅ Esta versión del notebook ya **no** hardcodea la API key (usa
`google.colab.userdata`) — bien hecho por el equipo, a diferencia de la
entrega anterior.

⚠️ **SDK deprecado**: `google.generativeai` (lo que usa el notebook)
ya no recibe actualizaciones de Google, que migró a un SDK unificado
nuevo (`google-genai`). Se dejó el código tal cual lo probó el equipo,
porque es lo único que confirmamos que funciona de verdad — yo no
tengo acceso a la API de Gemini desde mi entorno para probar una
migración. Si en algún momento deja de funcionar, la migración es
mecánica (ver `utils/audit_agent.py`).

## 6. Identidad visual

- **Nombre:** DEVERITAS
- **Logo:** escudo + ojo + red neuronal + checkmark como pupila —
  diseño propio en SVG (sin stickers ni íconos de stock), legible
  incluso a 32×32.
- **Paleta:** fondo casi negro (#0B0E14), acentos cian (#5EEAD4) y
  violeta (#8B7CF6), rojo coral para "rechazado" (#F2545B).
- **Tipografía:** Space Grotesk (titulares), Inter (cuerpo), JetBrains
  Mono (scores, session id, consola de "generando...") — la mono
  refuerza la idea de "lectura de instrumento forense" en vez de ser
  solo decorativa.
- **Iconografía de aceptar/rechazar:** diseño propio (círculo +
  check / círculo + diagonal), no corazón ni X genéricos.

## 7. Decisiones de diseño a propósito

- **Botones en vez de swipe con gestos:** se evaluó un componente
  bidireccional custom (drag real con JS) pero se priorizó
  confiabilidad — los botones nativos de Streamlit, restyleados para
  verse como controles circulares propios, no dependen de un build de
  componente separado y no se rompen entre versiones de Streamlit.
  El prototipo HTML (`deveritas_prototype.html`, fuera de esta carpeta)
  sí implementa swipe con gesto + teclado, por si más adelante quieren
  migrar esa interacción a un componente custom de Streamlit.
- **Resumen basado en reglas por default:** no existe hoy un modelo
  único pre-entrenado para este problema específico, así que el
  resumen por reglas garantiza que la parte 4 del flujo nunca falle
  mientras se integra el LLM real.
- **CSS inyectado en cada rerun:** Streamlit reconstruye el árbol de
  elementos completo en cada interacción, así que el `<style>` se
  re-emite siempre (cachear "ya se inyectó" hace que desaparezca en el
  segundo rerun).
- **Un solo bloque HTML por actualización en pantallas animadas:**
  `components/generator.py` y `components/report.py` (las pantallas con
  `time.sleep()` + `placeholder.markdown()` en loop) emiten *un único*
  `<div>` contenedor por actualización. Concatenar varias filas
  (`<div>...</div><div>...</div>`) en un mismo `st.markdown()` puede
  hacer que Streamlit renderice la primera bien y deje el resto como
  texto crudo sin parsear — si agregas más pantallas animadas, conviene
  mantener esa misma regla.
- **Pantalla de "generando" con la imagen del usuario:** en vez de un
  log de texto, se muestra la imagen cargada con una línea de escaneo
  (CSS puro) y una barra de progreso superpuesta — más amigable y
  coherente con el concepto de "laboratorio forense".
