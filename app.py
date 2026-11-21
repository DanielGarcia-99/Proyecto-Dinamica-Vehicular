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
        color: #1e293b;
        margin: 0;
    }
    .sandbox-pill {
        background-color: #e2e8f0;
        color: #475569;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #86efac;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #93c5fd;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #fca5a5;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #fcd34d;
    }
    .telemetry-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .telemetry-label {
        font-size: 0.74rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
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

    # Silueta aerodinámica estilizada del vehículo (orientado hacia la izquierda: frente en x=0, trasera en x=L)
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

    # Parabrisas y medallón trasero
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

    # Ejes delantero y trasero
    ax.plot([0, 0], [-W/2, W/2], color="#64748b", lw=2.0, zorder=3)
    ax.plot([L, L], [-W/2, W/2], color="#64748b", lw=2.0, zorder=3)
    ax.plot([0, L], [0, 0], color="#94a3b8", lw=1.0, ls="--", zorder=3)

    # Función auxiliar para dibujar cada neumático con ángulo
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

    # Ruedas traseras
    dibujar_rueda(L, W/2, 0)
    dibujar_rueda(L, -W/2, 0)
    # Ruedas delanteras (con giro Ackermann si aplica)
    dibujar_rueda(0, W/2, delta_in_deg, "#1e3a8a" if modo == "ackermann" else "#334155")
    dibujar_rueda(0, -W/2, delta_out_deg, "#065f46" if modo == "ackermann" else "#334155")

    if modo == "bicicleta":
        # Centro de Gravedad (CG) estilo símbolo ingeniería
        ax.plot(a, 0, 'o', markersize=14, markerfacecolor="#ffffff", markeredgecolor="#0f172a", markeredgewidth=2, zorder=7)
        ax.plot(a, 0, 'o', markersize=6, color="#ef4444", zorder=8)
        ax.text(a, 0.24, "CG", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=9)

        # Cotas a y b
        y_cota = -ancho_auto - 0.32
        ax.annotate("", xy=(0, y_cota), xytext=(a, y_cota), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.2))
        ax.annotate("", xy=(a, y_cota), xytext=(L, y_cota), arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.2))
        ax.text(a/2, y_cota - 0.08, f"a = {a:.2f} m", ha="center", va="top", fontsize=8.5, color="#334155", fontweight="bold")
        ax.text(a + b/2, y_cota - 0.08, f"b = {b:.2f} m", ha="center", va="top", fontsize=8.5, color="#334155", fontweight="bold")

        # Vectores de fuerza lateral Fyf y Fyr
        ax.annotate("", xy=(0, W/2 + 0.55), xytext=(0, W/2 + 0.12), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.2))
        ax.text(0.18, W/2 + 0.35, f"Fyf ({porcentaje_peso_f:.0f}%)", fontsize=8.5, color="#1d4ed8", fontweight="bold")

        ax.annotate("", xy=(L, W/2 + 0.55), xytext=(L, W/2 + 0.12), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.2))
        ax.text(L - 0.65, W/2 + 0.35, f"Fyr ({100-porcentaje_peso_f:.0f}%)", fontsize=8.5, color="#b91c1c", fontweight="bold")

        ax.text(0, 0, "Frente", ha="left", va="center", fontsize=8, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9), zorder=6)
        ax.text(L, 0, "Atrás", ha="right", va="center", fontsize=8, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.9), zorder=6)
    else:
        # Etiquetas de ángulo Ackermann en ruedas delanteras
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
        "Pulsa primero **🚗 Auto Familiar (Peso Adelante)** y observa el vector `CG` y la curva azul. "
        "Luego pulsa **⚠️ Auto Nervioso (Peso Atrás)**. ¿Qué cambia en la gráfica y en el badge de estado?"
    ),
    "Misión 3 (Dirección): El secreto para dar vuelta sin arrastrar llantas": (
        "**[Paso 1 - Observación]** Ve a **3. Dirección Ackermann** y sube el ángulo de giro a `35°`. "
        "Observa las dos ruedas delanteras en el chasis: ¿giran el mismo ángulo o la rueda interna gira más cerrada que la externa?"
    ),
    "Misión 4 (Seguridad): El reto de frenado en lluvia (ABS)": (
        "**[Paso 1 - Reto Práctico]** Ve a **4. Frenado ABS**, selecciona **🌧️ Pavimento Mojado**, "
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
        h1, h2 = st.columns(
