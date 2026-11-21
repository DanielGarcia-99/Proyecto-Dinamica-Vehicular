import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS (INTERACTIVE SANDBOX THEME)
# ==============================================================================
st.set_page_config(
    page_title="Vehicle Dynamics Sandbox | Tutor IA",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f9;
        color: #0f172a;
    }
    /* Forzar contraste oscuro en textos, pestañas, labels y subtítulos */
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #0f172a !important;
    }
    /* Mantener texto blanco dentro de la barra lateral oscura */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    /* Botones claros con texto oscuro legible */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
    }
    .sandbox-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.2rem;
    }
    .sandbox-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0;
    }
    .sandbox-pill {
        background-color: #e2e8f0;
        color: #334155 !important;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534 !important;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #86efac;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af !important;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #93c5fd;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b !important;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #fca5a5;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e !important;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #fcd34d;
    }
    .telemetry-card {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        border-top: 1px solid #cbd5e1;
        border-right: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
    }
    .telemetry-label {
        font-size: 0.74rem;
        color: #475569 !important;
        font-weight: 700;
        text-transform: uppercase;
    }
    .telemetry-val {
        font-size: 1.15rem;
        color: #0f172a !important;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)
    .telemetry-val {
        font-size: 1.15rem;
        color: #0f172a;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9

def obtener_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

API_KEY = obtener_api_key()

# ==============================================================================
# MOTORES MATEMÁTICOS DE NEUMÁTICOS (PACEJKA, JULIEN Y DUGOFF)
# ==============================================================================
def calcular_pacejka_long(kappa, B, C, D, E):
    Bx = B * kappa
    return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

def calcular_pacejka_lat(alpha_deg, B, C, D, E):
    alpha_rad = np.deg2rad(alpha_deg)
    Bx = B * alpha_rad
    return D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

def calcular_julien_long(kappa, Fz, mu, C_kappa):
    kappa_safe = np.maximum(np.abs(kappa), 1e-6)
    signo = np.sign(kappa)
    lambda_j = (mu * Fz) / (2.0 * C_kappa * kappa_safe)
    Fx = np.where(
        lambda_j >= 1.0,
        C_kappa * kappa_safe,
        mu * Fz * (1.0 - (mu * Fz) / (4.0 * C_kappa * kappa_safe))
    )
    return signo * Fx

def calcular_julien_lat(alpha_deg, Fz, mu, C_alpha):
    alpha_rad = np.deg2rad(alpha_deg)
    tan_a = np.maximum(np.abs(np.tan(alpha_rad)), 1e-6)
    signo = np.sign(alpha_rad)
    lambda_j = (mu * Fz) / (2.0 * C_alpha * tan_a)
    Fy = np.where(
        lambda_j >= 1.0,
        C_alpha * tan_a,
        mu * Fz * (1.0 - (mu * Fz) / (4.0 * C_alpha * tan_a))
    )
    return signo * Fy

def calcular_dugoff_long(kappa, Fz, mu, C_kappa):
    kappa_safe = np.minimum(np.abs(kappa), 0.999)
    signo = np.sign(kappa)
    denom = np.maximum(2.0 * C_kappa * kappa_safe, 1e-6)
    lam = (mu * Fz * (1.0 - kappa_safe)) / denom
    f_lam = np.where(lam < 1.0, (2.0 - lam) * lam, 1.0)
    Fx = (C_kappa * kappa_safe / (1.0 - kappa_safe)) * f_lam
    return signo * Fx

def calcular_dugoff_lat(alpha_deg, Fz, mu, C_alpha):
    alpha_rad = np.deg2rad(alpha_deg)
    tan_a = np.maximum(np.abs(np.tan(alpha_rad)), 1e-6)
    signo = np.sign(alpha_rad)
    lam = (mu * Fz) / (2.0 * C_alpha * tan_a)
    f_lam = np.where(lam < 1.0, (2.0 - lam) * lam, 1.0)
    Fy = C_alpha * tan_a * f_lam
    return signo * Fy

# ==============================================================================
# DIBUJO TÉCNICO EN VISTA SUPERIOR DEL VEHÍCULO (ESTILO SANDBOX CAD)
# ==============================================================================
def dibujar_chasis_top_view(ax, a, b, delta_in_deg=0.0, delta_out_deg=0.0, modo="bicicleta", porcentaje_peso_f=50.0):
    L = a + b
    W = 1.65
    ax.clear()

    x_frente = -0.55
    x_trasera = L + 0.55
    ancho_auto = W * 0.54

    carroceria = patches.FancyBboxPatch(
        (x_frente, -ancho_auto),
        x_trasera - x_frente,
        2 * ancho_auto,
        boxstyle="round,pad=0.08,rounding_size=0.38",
        linewidth=1.8,
        edgecolor="#475569",
        facecolor="#f1f5f9",
        zorder=2
    )
    ax.add_patch(carroceria)

    parabrisas = patches.Polygon(
        [[0.25, -ancho_auto*0.78], [0.75, -ancho_auto*0.65], [0.75, ancho_auto*0.65], [0.25, ancho_auto*0.78]],
        closed=True, facecolor="#cbd5e1", edgecolor="#64748b", lw=1.1, zorder=3
    )
    medallon = patches.Polygon(
        [[L-0.55, -ancho_auto*0.65], [L-0.15, -ancho_auto*0.75], [L-0.15, ancho_auto*0.75], [L-0.55, ancho_auto*0.65]],
        closed=True, facecolor="#cbd5e1", edgecolor="#64748b", lw=1.1, zorder=3
    )
    ax.add_patch(parabrisas)
    ax.add_patch(medallon)

    ax.plot([0, 0], [-W/2, W/2], color="#64748b", lw=2.0, zorder=3)
    ax.plot([L, L], [-W/2, W/2], color="#64748b", lw=2.0, zorder=3)
    ax.plot([0, L], [0, 0], color="#94a3b8", lw=1.0, ls="--", zorder=3)

    def dibujar_rueda(cx, cy, ang_deg, color_fill="#334155"):
        rw, rh = 0.54, 0.22
        rect = patches.FancyBboxPatch(
            (-rw/2, -rh/2), rw, rh,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color_fill, edgecolor="#0f172a", lw=1.2, zorder=5
        )
        t = plt.matplotlib.transforms.Affine2D().rotate_deg(ang_deg).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    dibujar_rueda(L, W/2, 0)
    dibujar_rueda(L, -W/2, 0)
    dibujar_rueda(0, W/2, delta_in_deg, "#1e3a8a" if modo == "ackermann" else "#334155")
    dibujar_rueda(0, -W/2, delta_out_deg, "#065f46" if modo == "ackermann" else "#334155")

    if modo == "bicicleta":
        ax.plot(a, 0, 'o', markersize=14, markerfacecolor="#ffffff", markeredgecolor="#0f172a", markeredgewidth=2, zorder=7)
        ax.plot(a, 0, 'o', markersize=6, color="#ef4444", zorder=8)
        ax.text(a, 0.24, "CG", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=9)

        y_cota = -ancho_auto - 0.32
        ax.annotate("", xy=(0, y_cota), xytext=(a, y_cota), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.2))
        ax.annotate("", xy=(a, y_cota), xytext=(L, y_cota), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.2))
        ax.text(a/2, y_cota - 0.08, f"a = {a:.2f} m", ha="center", va="top", fontsize=8.5, color="#334155", fontweight="bold")
        ax.text(a + b/2, y_cota - 0.08, f"b = {b:.2f} m", ha="center", va="top", fontsize=8.5, color="#334155", fontweight="bold")

        ax.annotate("", xy=(0, W/2 + 0.55), xytext=(0, W/2 + 0.12), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.2))
        ax.text(0.18, W/2 + 0.35, f"Fyf ({porcentaje_peso_f:.0f}%)", fontsize=8.5, color="#1d4ed8", fontweight="bold")

        ax.annotate("", xy=(L, W/2 + 0.55), xytext=(L, W/2 + 0.12), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.2))
        ax.text(L - 0.65, W/2 + 0.35, f"Fyr ({100-porcentaje_peso_f:.0f}%)", fontsize=8.5, color="#b91c1c", fontweight="bold")

        ax.text(0, 0, "Frente", ha="left", va="center", fontsize=8, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9), zorder=6)
        ax.text(L, 0, "Atrás", ha="right", va="center", fontsize=8, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9), zorder=6)
    else:
        ax.text(-0.15, W/2 + 0.35, f"δ_in = {delta_in_deg:.1f}°", ha="center", fontsize=9, color="#1e3a8a", fontweight="bold")
        ax.text(-0.15, -W/2 - 0.42, f"δ_out = {delta_out_deg:.1f}°", ha="center", fontsize=9, color="#065f46", fontweight="bold")

    ax.set_xlim(x_frente - 0.35, x_trasera + 0.35)
    ax.set_ylim(-W/2 - 0.75, W/2 + 0.78)
    ax.set_aspect("equal")
    ax.axis("off")

# ==============================================================================
# LÓGICA DEL TUTOR INTELIGENTE ADAPTATIVO (KOLB + CRAWL-WALK-RUN)
# ==============================================================================
MISIONES_PEDAGOGICAS = {
    "Libre (Exploración abierta)": (
        "¡Bienvenido al **Vehicle Dynamics Sandbox**! Explora libremente los paneles interactivos, mueve los controles "
        "en tiempo real y pregúntame qué significa cualquier curva o indicador de telemetría."
    ),
    "Misión 1 (Neumáticos): ¿Por qué patinan las llantas?": (
        "**[Paso 1 - Tu Hipótesis]** Antes de mover nada: si frenas o aceleras muy brusco, "
        "¿crees que la llanta agarra más mientras más patina, o llega un punto donde pierde agarre? "
        "Respóndeme lo que intuyes y luego compruébalo en la pestaña **1. Agarre de Llantas**."
    ),
    "Misión 2 (Manejo): Auto seguro vs. Auto nervioso (Subviraje/Sobreviraje)": (
        "**[Paso 1 - Experimento]** Ve a la pestaña **2. Manejo y Estabilidad**. "
        "Pulsa primero **🚗 Auto Familiar (Seguro)** y observa el vector `CG` y la curva azul. "
        "Luego pulsa **⚠️ Nervioso (Peso Atrás)**. ¿Qué cambia en la gráfica y en el badge de estado?"
    ),
    "Misión 3 (Dirección): El secreto para dar vuelta sin arrastrar llantas": (
        "**[Paso 1 - Observación]** Ve a **3. Dirección Ackermann** y sube el ángulo de giro a `35°`. "
        "Observa las dos ruedas delanteras en el chasis: ¿giran el mismo ángulo o la rueda interna gira más cerrada que la externa?"
    ),
    "Misión 4 (Seguridad): El reto de frenado en lluvia (ABS)": (
        "**[Paso 1 - Reto Práctico]** Ve a **4. Frenado ABS**, selecciona **🌧️ Lluvia (100 km/h)**, "
        "prueba primero con **ABS Apagado** y luego activa **ABS Encendido**. ¿Cuántos metros ahorraste?"
    )
}

def construir_prompt_pedagogico(nivel_usuario, mision_actual, contexto_simulacion):
    reglas_nivel = {
        "🟢 Explorador (Sin ingeniería)": (
            "NIVEL EXPLORADOR (CERO INGENIERÍA): Usa lenguaje 100% cotidiano, empático y visual. "
            "PROHIBIDO usar fórmulas matemáticas o jerga técnica sin traducirla al instante con una analogía diaria. "
            "Responde en máximo 2-3 párrafos cortos y termina con una pregunta sencilla."
        ),
        "🟡 Estudiante (Fundamentos)": (
            "NIVEL ESTUDIANTE: Combina intuición física con terminología correcta de dinámica vehicular "
            "(deslizamiento longitudinal, ángulo de deriva, subviraje, sobreviraje, transferencia de carga)."
        ),
        "🔴 Ingeniero (Análisis técnico)": (
            "NIVEL INGENIERO: Usa rigor técnico completo. Analiza las diferencias matemáticas entre Pacejka, Julien y Dugoff, "
            "rigidez de viraje (C_alpha), gradiente de subviraje (K_us) y estabilidad dinámica."
        )
    }
    return f"""Eres un Tutor Inteligente de Dinámica Vehicular basado en el Ciclo de Kolb, la progresión 'Understand-Crawl-Walk-Run' y aprendizaje por indagación.
{reglas_nivel[nivel_usuario]}
MISIÓN ACTIVA: {mision_actual}
TELEMETRÍA ACTUAL DEL SANDBOX: {contexto_simulacion}
Guía paso a paso sin dar la respuesta final de golpe y haz referencia directa a los valores que el usuario ve en pantalla."""

def tutor_respaldo_local(nivel_usuario, contexto_simulacion):
    return (
        f"📊 **Lectura en vivo del Sandbox ({nivel_usuario.split()[1]}):**\n\n"
        f"> `{contexto_simulacion}`\n\n"
        f"Observa cómo reaccionan las curvas y los indicadores de telemetría al mover los controles. ¿Qué patrón notas?"
    )

def consultar_gemini(mensaje_usuario, historial, nivel_usuario, mision_actual, contexto_simulacion):
    if not API_KEY:
        return tutor_respaldo_local(nivel_usuario, contexto_simulacion)
    try:
        client = genai.Client(api_key=API_KEY)
        instruccion = construir_prompt_pedagogico(nivel_usuario, mision_actual, contexto_simulacion)
        contenidos = []
        for msg in historial[-8:]:
            rol = "model" if msg["role"] == "assistant" else "user"
            contenidos.append(types.Content(role=rol, parts=[types.Part.from_text(text=msg["content"])]))
        contenidos.append(types.Content(role="user", parts=[types.Part.from_text(text=mensaje_usuario)]))

        respuesta = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contenidos,
            config=types.GenerateContentConfig(system_instruction=instruccion)
        )
        return respuesta.text
    except Exception:
        return tutor_respaldo_local(nivel_usuario, contexto_simulacion)

# ==============================================================================
# ESTADO DE SESIÓN Y BARRA LATERAL (TUTOR IA)
# ==============================================================================
if "telemetria" not in st.session_state:
    st.session_state.telemetria = "Sandbox inicializado."
if "mision_previa" not in st.session_state:
    st.session_state.mision_previa = "Misión 1 (Neumáticos): ¿Por qué patinan las llantas?"
if "nivel_previo" not in st.session_state:
    st.session_state.nivel_previo = "🟢 Explorador (Sin ingeniería)"
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = [{"role": "assistant", "content": MISIONES_PEDAGOGICAS[st.session_state.mision_previa]}]

with st.sidebar:
    st.header("🎓 Tutor Inteligente (IA)")
    nivel_usuario = st.radio(
        "1. Nivel del usuario:",
        ["🟢 Explorador (Sin ingeniería)", "🟡 Estudiante (Fundamentos)", "🔴 Ingeniero (Análisis técnico)"],
        index=0
    )
    mision_actual = st.selectbox(
        "2. Misión Guiada:",
        list(MISIONES_PEDAGOGICAS.keys()),
        index=1
    )

    if mision_actual != st.session_state.mision_previa or nivel_usuario != st.session_state.nivel_previo:
        st.session_state.mision_previa = mision_actual
        st.session_state.nivel_previo = nivel_usuario
        st.session_state.mensajes_chat = [{"role": "assistant", "content": MISIONES_PEDAGOGICAS[mision_actual]}]

    st.divider()
    contenedor_chat = st.container(height=370)
    with contenedor_chat:
        for msg in st.session_state.mensajes_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    pregunta = st.chat_input("Escribe tu hipótesis o pregunta...")
    if pregunta:
        st.session_state.mensajes_chat.append({"role": "user", "content": pregunta})
        with contenedor_chat:
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"):
                with st.spinner("Analizando telemetría..."):
                    resp = consultar_gemini(pregunta, st.session_state.mensajes_chat[:-1], nivel_usuario, mision_actual, st.session_state.telemetria)
                    st.markdown(resp)
        st.session_state.mensajes_chat.append({"role": "assistant", "content": resp})

# ==============================================================================
# ENCABEZADO PRINCIPAL ESTILO SANDBOX
# ==============================================================================
st.markdown("""
<div class="sandbox-header">
    <h1 class="sandbox-title">🏎️ Vehicle Dynamics Simulator</h1>
    <span class="sandbox-pill">Interactive Sandbox</span>
</div>
""", unsafe_allow_html=True)
st.caption("Laboratorio interactivo de dinámica vehicular con telemetría en tiempo real y Tutor IA adaptativo.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Agarre de Llantas",
    "2. Manejo y Estabilidad",
    "3. Dirección Ackermann",
    "4. Frenado ABS",
    "5. Fórmulas Matemáticas"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: NEUMÁTICOS (TARJETA SANDBOX)
# ------------------------------------------------------------------------------
with tab1:
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown("### Tire Mechanics & Grip Saturation")
        h2.markdown('<div style="text-align:right;"><span class="badge-blue">Pacejka · Julien · Dugoff</span></div>', unsafe_allow_html=True)

        if "t1_fz" not in st.session_state:
            st.session_state.t1_fz = 4000
            st.session_state.t1_mu = 0.90

        col_ctrl, col_plots, col_stat = st.columns([1.15, 3.2, 0.95])

        with col_ctrl:
            st.markdown("**Escenarios Rápidos**")
            if st.button("☀️ Asfalto Seco", use_container_width=True):
                st.session_state.t1_mu = 0.95
            if st.button("🌧️ Pavimento Mojado", use_container_width=True):
                st.session_state.t1_mu = 0.50
            if st.button("❄️ Hielo / Nieve", use_container_width=True):
                st.session_state.t1_mu = 0.20

            st.divider()
            Fz = st.slider("Carga Vertical Fz [N]", 1500, 8000, key="t1_fz", step=250)
            mu = st.slider("Coef. Fricción (μ)", 0.15, 1.20, key="t1_mu", step=0.05)
            modelos_visibles = st.multiselect(
                "Modelos activos:",
                ["Pacejka", "Julien", "Dugoff"],
                default=["Pacejka", "Julien", "Dugoff"]
            )

            with st.expander("⚙️ Rigidez y Pacejka"):
                C_long_kN = st.slider("C_κ [kN]", 30, 150, 80, 5)
                C_lat_kN = st.slider("C_α [kN/rad]", 20, 120, 60, 5)
                B_pac = st.slider("B (Rigidez)", 4.0, 20.0, 10.0, 0.5)
                C_pac = st.slider("C (Forma)", 1.1, 2.2, 1.65, 0.05)
                E_pac = st.slider("E (Curvatura)", -1.0, 1.0, 0.2, 0.1)

        C_kappa = C_long_kN * 1000.0
        C_alpha = C_lat_kN * 1000.0
        D_max = mu * Fz
        kappa = np.linspace(0, 1.0, 300)
        alpha = np.linspace(0, 20.0, 300)

        with col_plots:
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.1))
            if "Pacejka" in modelos_visibles:
                ax1.plot(kappa * 100, calcular_pacejka_long(kappa, B_pac, C_pac, D_max, E_pac), label='Pacejka', color='#1d4ed8', lw=2.3)
                ax2.plot(alpha, calcular_pacejka_lat(alpha, B_pac, C_pac, D_max, E_pac), label='Pacejka', color='#1d4ed8', lw=2.3)
            if "Julien" in modelos_visibles:
                ax1.plot(kappa * 100, calcular_julien_long(kappa, Fz, mu, C_kappa), label='Julien', color='#d97706', lw=2.0, ls='--')
                ax2.plot(alpha, calcular_julien_lat(alpha, Fz, mu, C_alpha), label='Julien', color='#d97706', lw=2.0, ls='--')
            if "Dugoff" in modelos_visibles:
                ax1.plot(kappa * 100, calcular_dugoff_long(kappa, Fz, mu, C_kappa), label='Dugoff', color='#059669', lw=2.0, ls='-.')
                ax2.plot(alpha, calcular_dugoff_lat(alpha, Fz, mu, C_alpha), label='Dugoff', color='#059669', lw=2.0, ls='-.')

            ax1.axhline(D_max, color='#64748b', ls=':', lw=1.1, label=f'Límite ({D_max:.0f} N)')
            ax1.set_title('Longitudinal Force Fx vs. Slip κ', fontsize=10, fontweight='bold', color='#1e293b')
            ax1.set_xlabel('Slip Ratio κ [%]', fontsize=9)
            ax1.set_ylabel('Fx [N]', fontsize=9)
            ax1.grid(True, ls='--', alpha=0.45)
            ax1.legend(fontsize=8)

            ax2.axhline(D_max, color='#64748b', ls=':', lw=1.1)
            ax2.set_title('Lateral Force Fy vs. Slip Angle α', fontsize=10, fontweight='bold', color='#1e293b')
            ax2.set_xlabel('Slip Angle α [deg]', fontsize=9)
            ax2.set_ylabel('Fy [N]', fontsize=9)
            ax2.grid(True, ls='--', alpha=0.45)
            if modelos_visibles:
                ax2.legend(fontsize=8)

            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with col_stat:
            st.markdown("**State / Telemetry**")
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Max Grip (μFz)</div><div class="telemetry-val">{D_max:.0f} N</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Friction CoF (μ)</div><div class="telemetry-val">{mu:.2f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Vertical Load</div><div class="telemetry-val">{Fz} N</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Cornering Stiff.</div><div class="telemetry-val">{C_lat_kN} kN/rad</div></div>', unsafe_allow_html=True)

    st.session_state.telemetria = f"[Neumáticos] Modelos={modelos_visibles}, Fz={Fz} N, mu={mu}, Fmax={D_max:.0f} N."

# ------------------------------------------------------------------------------
# PESTAÑA 2: MODELO BICICLETA (CON CHASIS CAD TOP-VIEW + CURVA SKIDPAD)
# ------------------------------------------------------------------------------
with tab2:
    if "t2_a" not in st.session_state:
        st.session_state.t2_a = 1.10
        st.session_state.t2_b = 1.60
        st.session_state.t2_v = 80

    with st.container(border=True):
        col_ctrl2, col_mid2, col_right2, col_stat2 = st.columns([1.15, 1.65, 1.65, 0.95])

        with col_ctrl2:
            st.markdown("**Presets de Vehículo**")
            if st.button("🚗 Auto Familiar (Seguro)", use_container_width=True):
                st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.05, 1.65, 80
            if st.button("🏎️ Deportivo (50/50)", use_container_width=True):
                st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.35, 1.35, 90
            if st.button("⚠️ Nervioso (Peso Atrás)", use_container_width=True):
                st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.70, 1.00, 95

            st.divider()
            a = st.slider("Dist. CG Delante (a) [m]", 0.70, 2.00, key="t2_a", step=0.05)
            b = st.slider("Dist. CG Atrás (b) [m]", 0.70, 2.00, key="t2_b", step=0.05)
            vel_kmh = st.slider("Velocidad [km/h]", 20, 160, key="t2_v", step=5)

            with st.expander("⚙️ Masa y Rigidez"):
                masa = st.slider("Masa [kg]", 800, 2500, 1500, 50)
                Caf_kN = st.slider("C_αf [kN/rad]", 30, 120, 65, 5)
                Car_kN = st.slider("C_αr [kN/rad]", 30, 120, 65, 5)

        L = a + b
        Wf = (masa * 9.81 * b) / L
        Wr = (masa * 9.81 * a) / L
        Kus = (Wf / (Caf_kN * 1000.0)) - (Wr / (Car_kN * 1000.0))
        Kus_deg_g = np.rad2deg(Kus * 9.81)
        pct_front = (b / L) * 100.0

        if Kus_deg_g > 0.3:
            badge_html = '<span class="badge-blue">Status: Understeer (Subviraje)</span>'
            estado_nombre = "Understeer"
        elif Kus_deg_g < -0.3:
            badge_html = '<span class="badge-red">Status: Oversteer (Sobreviraje)</span>'
            estado_nombre = "Oversteer"
        else:
            badge_html = '<span class="badge-green">Status: Neutral Steer</span>'
            estado_nombre = "Neutral"

        st.markdown(f"### Handling Dynamics & Mass Distribution &nbsp; {badge_html}", unsafe_allow_html=True)

        with col_mid2:
            st.markdown("**2-DoF Bicycle Model (Top View)**")
            fig_car, ax_car = plt.subplots(figsize=(5.2, 3.8))
            dibujar_chasis_top_view(ax_car, a, b, modo="bicicleta", porcentaje_peso_f=pct_front)
            plt.tight_layout()
            st.pyplot(fig_car)
            plt.close(fig_car)

        with col_right2:
            st.markdown("**Steering Angle vs. Lateral Acceleration**")
            ay_g = np.linspace(0, 0.85, 100)
            R_curva = 50.0
            delta_ack_deg = np.rad2deg(L / R_curva)
            delta_total_deg = delta_ack_deg + Kus_deg_g * ay_g

            fig_skid, ax_skid = plt.subplots(figsize=(5.2, 3.8))
            color_c = '#1d4ed8' if Kus > 0.0005 else ('#dc2626' if Kus < -0.0005 else '#059669')
            ax_skid.plot(ay_g, delta_total_deg, color=color_c, lw=2.5, label=f'Actual ({estado_nombre})')
            ax_skid.axhline(delta_ack_deg, color='#64748b', ls='--', lw=1.3, label='Neutral (Ackermann)')
            ax_skid.set_xlabel('Lateral Acceleration [g]', fontsize=9)
            ax_skid.set_ylabel('Steering Angle [deg]', fontsize=9)
            ax_skid.grid(True, ls='--', alpha=0.45)
            ax_skid.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_skid)
            plt.close(fig_skid)

        with col_stat2:
            st.markdown("**State**")
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Speed</div><div class="telemetry-val">{vel_kmh} km/h</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Kus Gradient</div><div class="telemetry-val">{Kus_deg_g:+.2f} °/g</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Weight Dist.</div><div class="telemetry-val">{pct_front:.0f}/{100-pct_front:.0f}</div></div>', unsafe_allow_html=True)
            if Kus < -0.0005:
                v_crit = np.sqrt((-L) / Kus) * 3.6
                st.markdown(f'<div class="telemetry-card" style="border-left-color:#dc2626;"><div class="telemetry-label">V. Crítica</div><div class="telemetry-val">{v_crit:.0f} km/h</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Wheelbase L</div><div class="telemetry-val">{L:.2f} m</div></div>', unsafe_allow_html=True)

    st.session_state.telemetria = f"[Modelo Bicicleta] Reparto={pct_front:.0f}%/{100-pct_front:.0f}%, Kus={Kus_deg_g:.2f} deg/g, Estado={estado_nombre}, V={vel_kmh} km/h."

# ------------------------------------------------------------------------------
# PESTAÑA 3: DIRECCIÓN ACKERMANN (CON CHASIS ORIENTABLE)
# ------------------------------------------------------------------------------
with tab3:
    with st.container(border=True):
        st.markdown('### Ackermann Steering Geometry &nbsp; <span class="badge-green">Kinematic Turning</span>', unsafe_allow_html=True)
        c3_ctrl, c3_car, c3_plot, c3_stat = st.columns([1.15, 1.65, 1.65, 0.95])

        with c3_ctrl:
            delta_in_deg = st.slider("Giro Rueda Interna [°]", 0, 45, 25, 1)
            porcentaje_ack = st.slider("% Ackermann", 0, 120, 100, 10)
            L_ack = st.slider("Batalla L [m]", 1.8, 3.5, 2.7, 0.1)
            W_ack = st.slider("Trocha W [m]", 1.2, 2.0, 1.6, 0.05)

        delta_in = np.deg2rad(delta_in_deg)
        if delta_in_deg < 0.5:
            delta_out_ideal_deg = float(delta_in_deg)
            R_giro = 999.0
        else:
            R_giro = L_ack / np.tan(delta_in) + W_ack / 2.0
            delta_out_ideal_deg = np.rad2deg(np.arctan(L_ack / (R_giro + W_ack / 2.0)))

        factor = porcentaje_ack / 100.0
        delta_out_real_deg = delta_in_deg - factor * (delta_in_deg - delta_out_ideal_deg)
        diferencia_deg = delta_in_deg - delta_out_real_deg

        with c3_car:
            st.markdown("**Front Wheels Steer Angle**")
            fig_ack_car, ax_ack_car = plt.subplots(figsize=(5.2, 3.8))
            dibujar_chasis_top_view(ax_ack_car, L_ack * 0.45, L_ack * 0.55, delta_in_deg, delta_out_real_deg, modo="ackermann")
            plt.tight_layout()
            st.pyplot(fig_ack_car)
            plt.close(fig_ack_car)

        with c3_plot:
            st.markdown("**Inner vs. Outer Wheel Curve**")
            rango_in = np.linspace(0, 45, 100)
            rango_rad = np.deg2rad(rango_in)
            R_arr = np.where(rango_in > 0.1, L_ack / np.tan(np.maximum(rango_rad, 1e-4)) + W_ack / 2.0, 1e6)
            ideal_out = np.rad2deg(np.arctan(L_ack / (R_arr + W_ack / 2.0)))
            real_out = rango_in - factor * (rango_in - ideal_out)

            fig_ack_p, ax_ack_p = plt.subplots(figsize=(5.2, 3.8))
            ax_ack_p.plot(rango_in, rango_in, 'k--', lw=1.2, label='0% Paralelo')
            ax_ack_p.plot(rango_in, ideal_out, color='#1d4ed8', ls=':', lw=1.8, label='100% Ackermann')
            ax_ack_p.plot(rango_in, real_out, color='#059669', lw=2.3, label=f'Actual ({porcentaje_ack}%)')
            ax_ack_p.plot(delta_in_deg, delta_out_real_deg, 'ro', markersize=7)
            ax_ack_p.set_xlabel('Inner Wheel δ_in [deg]', fontsize=9)
            ax_ack_p.set_ylabel('Outer Wheel δ_out [deg]', fontsize=9)
            ax_ack_p.grid(True, ls='--', alpha=0.45)
            ax_ack_p.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_ack_p)
            plt.close(fig_ack_p)

        with c3_stat:
            st.markdown("**State**")
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Inner Angle</div><div class="telemetry-val">{delta_in_deg:.1f}°</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Outer Angle</div><div class="telemetry-val">{delta_out_real_deg:.1f}°</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Toe-Out Diff</div><div class="telemetry-val">{diferencia_deg:.1f}°</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Turn Radius</div><div class="telemetry-val">{min(R_giro, 99.9):.1f} m</div></div>', unsafe_allow_html=True)

    st.session_state.telemetria = f"[Ackermann] %Ack={porcentaje_ack}%, Interna={delta_in_deg:.1f}°, Externa={delta_out_real_deg:.1f}°."

# ------------------------------------------------------------------------------
# PESTAÑA 4: SIMULADOR DE FRENADO ABS (TARJETA SANDBOX CON BANDA ÓPTIMA)
# ------------------------------------------------------------------------------
def _correr_integracion_abs(modelo_nombre, abs_activo, v_inicial_kmh, mu_abs, torque_freno_max):
    m_quarter = 375.0
    Fz = m_quarter * 9.81
    Rw = 0.32
    Iw = 1.2
    C_kappa = 80000.0
    dt = 0.002
    pasos = int(7.0 / dt)

    V = v_inicial_kmh / 3.6
    omega = V / Rw
    hist_t, hist_V_kmh, hist_Vw_kmh, hist_kappa = [], [], [], []
    distancia = 0.0
    valvula_alivio = False
    tiempo_bloqueo = 0.0

    for step in range(pasos):
        t = step * dt
        if V <= 0.15:
            hist_t.append(t)
            hist_V_kmh.append(0.0)
            hist_Vw_kmh.append(0.0)
            hist_kappa.append(0.0)
            break

        v_rueda = omega * Rw
        kappa = np.clip((V - v_rueda) / max(V, 0.5), 0.0, 1.0)

        if modelo_nombre == "Pacejka":
            Fx = float(calcular_pacejka_long(np.array([kappa]), 10.0, 1.65, mu_abs * Fz, 0.2)[0])
        elif modelo_nombre == "Julien":
            Fx = float(calcular_julien_long(np.array([kappa]), Fz, mu_abs, C_kappa)[0])
        else:
            Fx = float(calcular_dugoff_long(np.array([kappa]), Fz, mu_abs, C_kappa)[0])

        if abs_activo and V > 1.67:
            if kappa > 0.18:
                valvula_alivio = True
            elif kappa < 0.12:
                valvula_alivio = False
            torque_aplicado = (0.10 * torque_freno_max) if valvula_alivio else torque_freno_max
        else:
            torque_aplicado = torque_freno_max

        if kappa >= 0.95 and V > 1.67:
            tiempo_bloqueo += dt

        V = max(0.0, V + (-Fx / m_quarter) * dt)
        omega = max(0.0, omega + ((Fx * Rw - torque_aplicado) / Iw) * dt)
        distancia += V * dt

        hist_t.append(t)
        hist_V_kmh.append(V * 3.6)
        hist_Vw_kmh.append(omega * Rw * 3.6)
        hist_kappa.append(kappa * 100.0)

    return hist_t, hist_V_kmh, hist_Vw_kmh, hist_kappa, distancia, hist_t[-1], tiempo_bloqueo, Fz

with tab4:
    if "t4_v" not in st.session_state:
        st.session_state.t4_v = 100
        st.session_state.t4_mu = 0.85
        st.session_state.t4_tq = 2400

    with st.container(border=True):
        c4_ctrl, c4_p1, c4_p2, c4_stat = st.columns([1.15, 1.65, 1.65, 0.95])

        with c4_ctrl:
            abs_activo = st.toggle("🛡️ ABS System Active", value=True)
            modelo_abs = st.selectbox("Tire Model:", ["Comparar los 3", "Pacejka", "Julien", "Dugoff"], index=0)
            st.divider()
            if st.button("☀️ Seco (100 km/h)", use_container_width=True):
                st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 100, 0.85, 2400
            if st.button("🌧️ Lluvia (100 km/h)", use_container_width=True):
                st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 100, 0.45, 2400
            if st.button("❄️ Hielo (80 km/h)", use_container_width=True):
                st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 80, 0.20, 2400

            v_ini_kmh = st.slider("Initial Speed [km/h]", 30, 160, key="t4_v", step=5)
            mu_abs = st.slider("Surface CoF (μ)", 0.15, 1.10, key="t4_mu", step=0.05)
            torque_max = st.slider("Brake Torque [N·m]", 500, 3500, key="t4_tq", step=100)

        badge_abs = '<span class="badge-amber">Optimal Braking Band (12% - 18%)</span>' if abs_activo else '<span class="badge-red">ABS OFF · Wheel Lock Risk</span>'
        st.markdown(f"### Emergency ABS Braking Simulation &nbsp; {badge_abs}", unsafe_allow_html=True)

        fig_v, ax_v = plt.subplots(figsize=(5.2, 3.8))
        fig_k, ax_k = plt.subplots(figsize=(5.2, 3.8))

        if modelo_abs == "Comparar los 3":
            estilos = [("Pacejka", "#1d4ed8", "-"), ("Julien", "#d97706", "--"), ("Dugoff", "#059669", "-.")]
            res_dict = {}
            for nombre, color, ls in estilos:
                ht, hV, hVw, hk, dist, t_det, t_bloq, Fz = _correr_integracion_abs(nombre, abs_activo, v_ini_kmh, mu_abs, torque_max)
                ax_v.plot(ht, hV, label=f'{nombre} ({dist:.1f} m)', color=color, lw=2.2, ls=ls)
                ax_k.plot(ht, hk, label=nombre, color=color, lw=1.9, ls=ls)
                res_dict[nombre] = (dist, t_det, t_bloq)

            with c4_stat:
                st.markdown("**Braking Distance**")
                for nombre in ["Pacejka", "Julien", "Dugoff"]:
                    d, t_d, t_b = res_dict[nombre]
                    st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">{nombre}</div><div class="telemetry-val">{d:.2f} m</div><div style="font-size:0.72rem;color:#64748b;">{t_d:.2f} s · Lock: {t_b:.2f} s</div></div>', unsafe_allow_html=True)

            st.session_state.telemetria = f"[ABS Comparativo] ABS={abs_activo}, V={v_ini_kmh} km/h, mu={mu_abs}, Pacejka={res_dict['Pacejka'][0]:.2f}m."
        else:
            ht, hV, hVw, hk, dist, t_det, t_bloq, Fz = _correr_integracion_abs(modelo_abs, abs_activo, v_ini_kmh, mu_abs, torque_max)
            ax_v.plot(ht, hV, label='Vehicle Speed', color='#1d4ed8', lw=2.4)
            ax_v.plot(ht, hVw, label='Wheel Speed', color='#dc2626', lw=1.8, ls='--')
            ax_k.plot(ht, hk, label='Tire Slip κ', color='#059669', lw=2.1)

            with c4_stat:
                st.markdown("**State**")
                st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Stop Distance</div><div class="telemetry-val">{dist:.2f} m</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Stop Time</div><div class="telemetry-val">{t_det:.2f} s</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="telemetry-card"><div class="telemetry-label">Wheel Lock</div><div class="telemetry-val">{t_bloq:.2f} s</div></div>', unsafe_allow_html=True)

            st.session_state.telemetria = f"[ABS {modelo_abs}] ABS={abs_activo}, V={v_ini_kmh} km/h, Dist={dist:.2f}m, Lock={t_bloq:.2f}s."

        ax_v.set_title('Braking Deceleration (Speed vs. Time)', fontsize=10, fontweight='bold')
        ax_v.set_xlabel('Time [s]', fontsize=9)
        ax_v.set_ylabel('Speed [km/h]', fontsize=9)
        ax_v.grid(True, ls='--', alpha=0.45)
        ax_v.legend(fontsize=8)

        ax_k.axhspan(12, 18, color='#facc15', alpha=0.28, label='Optimal Band (12%-18%)')
        ax_k.axhline(100, color='#dc2626', ls=':', lw=1.2, label='100% Lock')
        ax_k.set_title('Tire Slip Ratio (%)', fontsize=10, fontweight='bold')
        ax_k.set_xlabel('Time [s]', fontsize=9)
        ax_k.set_ylabel('Slip κ [%]', fontsize=9)
        ax_k.set_ylim(-5, 110)
        ax_k.grid(True, ls='--', alpha=0.45)
        ax_k.legend(fontsize=8)

        with c4_p1:
            plt.tight_layout()
            st.pyplot(fig_v)
            plt.close(fig_v)
        with c4_p2:
            plt.tight_layout()
            st.pyplot(fig_k)
            plt.close(fig_k)

# ------------------------------------------------------------------------------
# PESTAÑA 5: FORMULARIO MATEMÁTICO DE REFERENCIA
# ------------------------------------------------------------------------------
with tab5:
    with st.container(border=True):
        st.subheader("📐 Formulario Matemático del Simulador")
        st.markdown(r"""
        ### 1. Modelos Matemáticos del Neumático
        * **Deslizamiento longitudinal ($\kappa$):** $\kappa = \dfrac{V - \omega R_w}{\max(V, \, \omega R_w)}$
        * **Fórmula Mágica de Pacejka:**
          $$F(x) = D \sin\Big( C \arctan\big( B x - E [B x - \arctan(B x)] \big) \Big), \quad D = \mu F_z$$
        * **Modelo Teórico de Julien:**
          $$\lambda_J = \frac{\mu F_z}{2 C_\kappa |\kappa|}, \quad F_x = \begin{cases} C_\kappa \kappa & \lambda_J \ge 1 \\ \mu F_z \left(1 - \dfrac{\mu F_z}{4 C_\kappa |\kappa|}\right)\text{sgn}(\kappa) & \lambda_J < 1 \end{cases}$$
        * **Modelo Analítico de Dugoff:**
          $$\lambda = \frac{\mu F_z (1 - |\kappa|)}{2 C_\kappa |\kappa|}, \quad f(\lambda) = \begin{cases} (2 - \lambda)\lambda & \lambda < 1 \\ 1 & \lambda \ge 1 \end{cases}, \quad F_x = \frac{C_\kappa \kappa}{1 - |\kappa|} f(\lambda)$$

        ---

        ### 2. Modelo Bicicleta de 2 Grados de Libertad
        $$W_f = \frac{m g \, b}{L}, \quad W_r = \frac{m g \, a}{L}, \quad K_{us} = \frac{W_f}{C_{\alpha f}} - \frac{W_r}{C_{\alpha r}}, \quad \delta = \frac{L}{R} + K_{us} \, a_y$$
        * **Velocidad crítica de sobreviraje ($K_{us} < 0$):** $V_{\text{crítica}} = \sqrt{-L / K_{us}}$

        ---

        ### 3. Geometría de Dirección Ackermann
        $$\cot(\delta_{\text{out, ideal}}) - \cot(\delta_{\text{in}}) = \frac{W}{L}, \qquad \delta_{\text{out, real}} = \delta_{\text{in}} - p \, (\delta_{\text{in}} - \delta_{\text{out, ideal}})$$

        ---

        ### 4. Dinámica Longitudinal de Frenado y Control ABS
        $$m_{\text{cuarto}} \frac{dV}{dt} = -F_x(\kappa), \qquad I_w \frac{d\omega}{dt} = F_x(\kappa) R_w - T_{\text{freno}}(t)$$
        $$T_{\text{freno}}(t) = \begin{cases} 0.10 \, T_{\text{máx}} & \text{si } \kappa > 0.18 \text{ (válvula de alivio)} \\ T_{\text{máx}} & \text{si } \kappa < 0.12 \text{ (reaplicación)} \end{cases}$$
        """)
