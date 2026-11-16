"""UI Streamlit para consumir la API del BCMM Predictor — Edición corporativa."""
import html
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="BCMM Predictor · INEGI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- Estilos
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

    html, body, .stApp, .stMarkdown, [data-testid="stSidebar"],
    h1, h2, h3, h4, h5, h6, button, input, select, textarea, label, p, span {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    .stApp { background: #f8fafc; }

    /* Header corporativo */
    .brand-header {
        display: flex; align-items: center; gap: 1.1rem;
        padding: 1.4rem 0 1.2rem 0;
        border-bottom: 2px solid #0f172a;
        margin-bottom: 1.75rem;
    }
    .brand-mark {
        width: 56px; height: 56px; border-radius: 12px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.22);
        flex-shrink: 0;
    }
    .brand-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 700; font-size: 2.05rem; color: #0f172a;
        line-height: 1.15; margin: 0; letter-spacing: -0.015em;
    }
    .brand-subtitle {
        color: #64748b; font-size: 0.9rem; font-weight: 500; margin-top: 0.3rem;
    }
    .brand-tag {
        display: inline-block; background: #eff6ff; color: #1e40af;
        font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.55rem;
        border-radius: 999px; margin-left: 0.6rem; letter-spacing: 0.06em;
        vertical-align: middle;
    }

    /* Títulos de sección */
    .section-title {
        font-size: 0.74rem; font-weight: 700; color: #475569;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 1.6rem 0 0.7rem 0; padding-bottom: 0.45rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-title {
        font-size: 0.74rem; font-weight: 700; color: #475569;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin: 0.2rem 0 0.9rem 0;
    }
    .status-pill {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: #ecfdf5; color: #065f46;
        font-size: 0.8rem; font-weight: 600;
        padding: 0.35rem 0.75rem; border-radius: 999px;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18);
    }
    .status-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0; border-bottom: 1px solid #f1f5f9;
        font-size: 0.88rem;
    }
    .status-label { color: #64748b; font-weight: 500; }
    .status-value { color: #0f172a; font-weight: 600; text-align: right; }
    .endpoint-code {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important;
        font-size: 0.76rem; color: #334155;
        background: #f1f5f9; padding: 0.3rem 0.55rem;
        border-radius: 6px; word-break: break-all;
        display: block; margin-top: 0.4rem;
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 4px solid #1e40af; border-radius: 10px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .metric-label {
        font-size: 0.7rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.6rem; font-weight: 700; color: #0f172a;
        line-height: 1.15; font-variant-numeric: tabular-nums;
    }
    .metric-value--text { font-size: 1.1rem; font-weight: 600; word-break: break-word; }
    .metric-caption {
        font-size: 0.76rem; color: #94a3b8; margin-top: 0.35rem; font-weight: 500;
    }

    /* Botón principal */
    .stButton > button[kind="primary"] {
        background: #0f172a; color: #ffffff; border: none;
        border-radius: 8px; font-weight: 600; letter-spacing: 0.02em;
        padding: 0.7rem 1.3rem; transition: all 0.18s ease;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
    }
    .stButton > button[kind="primary"]:hover {
        background: #1e3a8a;
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.28);
        transform: translateY(-1px);
    }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div {
        border-radius: 8px;
    }

    /* Footer */
    .footer-note {
        margin-top: 3rem; padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.76rem; color: #94a3b8; text-align: center;
    }

    /* Limpieza de cromo default */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Header
st.markdown(
    """
<div class="brand-header">
    <div class="brand-mark">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 17L9 11L13 15L21 7" stroke="#ffffff" stroke-width="2.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M15 7H21V13" stroke="#ffffff" stroke-width="2.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    <div>
        <div class="brand-title">BCMM Predictor<span class="brand-tag">INEGI</span></div>
        <div class="brand-subtitle">
            Predicción del Comercio Exterior de México · Balanza Comercial de Mercancías
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- Cliente API
def api_get(path: str, **kwargs):
    """GET con manejo explícito de errores de conexión y JSON."""
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, timeout=10, **kwargs)
    except requests.exceptions.ConnectionError as e:
        st.error(
            f"**No se pudo conectar con la API en** `{url}`\n\n"
            f"¿Está corriendo Uvicorn? Ejecuta en otra terminal:\n\n"
            f"```\nuv run uvicorn api.main:app --reload --port 8000\n```\n\n"
            f"Detalle técnico: `{e.__class__.__name__}`"
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error(f"**Timeout** al llamar `{url}`. La API no respondió en 10 s.")
        st.stop()

    if r.status_code != 200:
        st.error(
            f"**La API respondió {r.status_code}** en `{url}`\n\n"
            f"**Body:**\n```\n{r.text[:500]}\n```"
        )
        st.stop()

    try:
        return r.json()
    except ValueError:
        st.error(
            f"**Respuesta 200 con JSON inválido** desde `{url}`\n\n"
            f"**Content-Type:** `{r.headers.get('content-type')}`\n\n"
            f"**Body (primeros 500 chars):**\n```\n{r.text[:500]}\n```\n\n"
            f"Verifica la URL base: `{API_URL}`"
        )
        st.stop()


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-title">Estado del servicio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="endpoint-code">{html.escape(API_URL)}</span>',
        unsafe_allow_html=True,
    )

    health = api_get("/health")
    st.markdown(
        f'<div class="status-pill" style="margin-top:0.9rem">'
        f'<span class="status-dot"></span>Operativo · {html.escape(str(health["status"]))}'
        f"</div>",
        unsafe_allow_html=True,
    )

    meta = api_get("/metadata")
    st.markdown(
        f"""
<div style="margin-top:1.2rem">
    <div class="status-row">
        <span class="status-label">Modelo</span>
        <span class="status-value">{html.escape(str(meta["model"]))}</span>
    </div>
    <div class="status-row">
        <span class="status-label">Versión</span>
        <span class="status-value">{html.escape(str(meta["version"]))}</span>
    </div>
    <div class="status-row">
        <span class="status-label">Países</span>
        <span class="status-value">{meta["n_paises"]}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Métricas de validación"):
        st.json(meta.get("metrics_val", {}))


# ---------------------------------------------------------------- Formulario
st.markdown(
    '<div class="section-title">Parámetros de la predicción</div>',
    unsafe_allow_html=True,
)

paises_data = api_get("/paises")

# Actualizamos la lista de opciones para mostrar el nombre real
opciones = [
    f"{p['pais_o_d']} — {p.get('nombre', p['pais_o_d'])}"
    for p in paises_data
]

default_idx = next((i for i, o in enumerate(opciones) if o.startswith("USA")), 0)

with st.container(border=True):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        pais_sel = st.selectbox("País (ISO-3)", opciones, index=default_idx)
        pais = pais_sel.split(" ")[0]
        tipo = st.selectbox("Flujo comercial", ["Exportaciones", "Importaciones"])
        tipo_bien = st.selectbox("Tipo de bien", ["Consumo", "Intermedio", "Capital"])

    with col2:
        anio = st.number_input("Año", min_value=2015, max_value=2030, value=2026, step=1)
        mes = st.slider("Mes", 1, 12, 3)

predict_clicked = st.button(
    "Generar predicción",
    type="primary",
    use_container_width=True,
)

# ---------------------------------------------------------------- Resultado
if predict_clicked:
    payload = {
        "pais_o_d": pais,
        "tipo": tipo,
        "tipo_bien": tipo_bien,
        "anio": int(anio),
        "mes": int(mes),
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error("La API se desconectó entre el GET inicial y el POST.")
        st.stop()

    if r.status_code != 200:
        st.error(f"**Error {r.status_code}:**\n```\n{r.text}\n```")
        st.stop()

    result = r.json()
    st.success("Predicción generada correctamente")

    st.markdown(
        '<div class="section-title">Resultado</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="medium")

    m1.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Valor predicho</div>
    <div class="metric-value">${result['predicted_val_usd_miles']:,.0f}</div>
    <div class="metric-caption">miles de USD</div>
</div>
""",
        unsafe_allow_html=True,
    )

    m2.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Modelo</div>
    <div class="metric-value metric-value--text">{html.escape(str(result["model"]))}</div>
    <div class="metric-caption">algoritmo</div>
</div>
""",
        unsafe_allow_html=True,
    )

    m3.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">Versión</div>
    <div class="metric-value metric-value--text">{html.escape(str(result["version"]))}</div>
    <div class="metric-caption">release</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if result.get("warning"):
        st.warning(result["warning"])

    with st.expander("Detalles técnicos"):
        st.json(result)

# ---------------------------------------------------------------- Footer
st.markdown(
    '<div class="footer-note">'
    "Fuente de datos: INEGI · Balanza Comercial de Mercancías de México (BCMM)"
    "</div>",
    unsafe_allow_html=True,
)