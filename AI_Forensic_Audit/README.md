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
├── app.py                  # orquesta el state machine de pantallas
├── requirements.txt        # streamlit + pillow + numpy (modo simulado)
├── requirements-models.txt # torch/diffusers/transformers (modelos reales)
├── download_models.py      # descarga un modelo abierto de prueba
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
│   ├── json_manager.py       # construye/lee el JSON de auditoría
│   ├── loader.py             # carga .safetensors/.pth/.pkl con fallback a mock
│   ├── inference.py          # generación + resumen (mock y hooks reales)
│   ├── svg.py                # logo, íconos propios, dial de score
│   └── styling.py            # inyección del CSS global
│
├── modelo_generativo/        # coloca aquí el modelo de la parte 2
├── modelo_resumen/           # coloca aquí el modelo de la parte 4
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

El que consume el agente de resumen (parte 4) — actualizado por el
equipo a nombres en inglés, con la métrica de tiempo de decisión:

```json
{
  "session_id": "audit_a1b2c3d4e5",
  "input_image": "inputs/source.jpg",
  "variations": [
    {
      "id": "V1",
      "image_path": "outputs/var_1.png",
      "authenticity_score": 0.86,
      "decision": "accepted",
      "decision_time_seconds": 5.8
    }
  ]
}
```

`decision` es `"accepted"` o `"rejected"`. `decision_time_seconds` se
mide desde que la variación aparece en pantalla hasta que el usuario
hace clic en aprobar/rechazar (`utils/session.py` + `components/cards.py`).

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
