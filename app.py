import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Laboratorio de Dinámica Vehicular",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

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
# LÓGICA DEL TUTOR INTELIGENTE ADAPTATIVO (KOLB + CRAWL-WALK-RUN)
# ==============================================================================
MISIONES_PEDAGOGICAS = {
    "Libre (Exploración abierta)": (
        "¡Hola! Explora libremente cualquiera de las pestañas. Usa los botones de escenarios rápidos o mueve los controles "
        "y pregúntame qué significa cualquier línea o comportamiento."
    ),
    "Misión 1 (Neumáticos): ¿Por qué patinan las llantas?": (
        "**[Paso 1 - Tu Hipótesis]** Antes de mover nada: si vas en un auto y frenas o aceleras muy brusco, "
        "¿crees que la llanta agarra más mientras más patina, o llega un punto donde pierde agarre? "
        "Respóndeme lo que intuyes y luego ve a la **Pestaña 1 (Neumáticos)** a comprobarlo."
    ),
    "Misión 2 (Manejo): Auto seguro vs. Auto nervioso (Subviraje/Sobreviraje)": (
        "**[Paso 1 - Experimento]** Ve a la **Pestaña 2 (Manejo en Curva)**. "
        "Haz clic primero en **🚗 Auto Familiar (Peso Adelante)** y observa la gráfica. "
        "Luego haz clic en **⚠️ Auto Nervioso (Peso Atrás)**. "
        "Cuéntame: ¿qué cambió en la gráfica y cuál crees que fabrican las marcas para uso diario?"
    ),
    "Misión 3 (Dirección): El secreto para dar vuelta sin arrastrar llantas": (
        "**[Paso 1 - Observación]** Ve a la **Pestaña 3 (Dirección Ackermann)** y sube el giro a `35°`. "
        "Mira las dos ruedas delanteras en el dibujo: ¿giran el mismo ángulo o la rueda de adentro gira más cerrada que la de afuera? "
        "Dime qué observas y por qué crees que pasa eso."
    ),
    "Misión 4 (Seguridad): El reto de frenado en lluvia (ABS)": (
        "**[Paso 1 - Reto Práctico]** Ve a la **Pestaña 4 (Frenado ABS)**. "
        "Haz clic en **🌧️ Pavimento Mojado**, prueba primero con **ABS Apagado** y luego activa **ABS Encendido**. "
        "Compárame cuántos metros ahorraste y qué diferencia ves entre los modelos de llanta."
    )
}

def construir_prompt_pedagogico(nivel_usuario, mision_actual, contexto_simulacion):
    reglas_nivel = {
        "🟢 Explorador (Sin ingeniería)": (
            "NIVEL EXPLORADOR (CERO INGENIERÍA): Usa lenguaje 100% cotidiano, empático y visual. "
            "PROHIBIDO usar fórmulas matemáticas, derivadas o jerga técnica sin traducirla al instante "
            "con una analogía de la vida diaria (ej. tenis en piso mojado, carrito de supermercado, bicicleta). "
            "Mantén tus respuestas breves (máximo 2-3 párrafos cortos) y termina siempre con una pregunta sencilla."
        ),
        "🟡 Estudiante (Fundamentos)": (
            "NIVEL ESTUDIANTE: Combina intuición física con terminología correcta de dinámica vehicular "
            "(deslizamiento longitudinal, ángulo de deriva, subviraje, sobreviraje, transferencia de carga). "
            "Explica cómo leer las gráficas (pendientes, picos y saturación) sin abrumar con ecuaciones largas."
        ),
        "🔴 Ingeniero (Análisis técnico)": (
            "NIVEL INGENIERO: Usa rigor técnico completo. Analiza las diferencias matemáticas y limitaciones "
            "entre Pacejka (Magic Formula), Julien y Dugoff, rigidez de viraje (C_alpha), gradiente de subviraje (K_us) "
            "y criterios de estabilidad dinámica."
        )
    }
    return f"""Eres un Tutor Inteligente de Dinámica Vehicular diseñado bajo tres principios pedagógicos:
1. Ciclo de Aprendizaje Experiencial de Kolb (Experiencia Concreta -> Observación Reflexiva -> Conceptualización Abstracta -> Experimentación Activa).
2. Progresión 'Understand-Crawl-Walk-Run' (construir sentido de ingeniería antes de confiar ciegamente en las gráficas).
3. Aprendizaje basado en indagación y retos (Serious Gaming: guiar al usuario a formular hipótesis y descubrir por sí mismo).

{reglas_nivel[nivel_usuario]}

MISIÓN ACTIVA DEL USUARIO: {mision_actual}

TELEMETRÍA ACTUAL DEL SIMULADOR (Lo que el usuario está viendo ahora en pantalla):
{contexto_simulacion}

INSTRUCCIONES DE TUTORÍA:
- Nunca des la respuesta final de golpe si el usuario está en una misión; guíalo paso a paso como un coach en pista.
- Haz referencia directa a los números o gráficas que el usuario tiene actualmente en su simulador.
- Fomenta el 'sentido de ingeniería': advierte cuando un parámetro elegido sea irreal o peligroso en la vida real.
"""

def tutor_respaldo_local(nivel_usuario, contexto_simulacion):
    intro = {
        "🟢 Explorador (Sin ingeniería)": "💡 **Análisis rápido del simulador:** ",
        "🟡 Estudiante (Fundamentos)": "📊 **Diagnóstico técnico en vivo:** ",
        "🔴 Ingeniero (Análisis técnico)": "⚙️ **Evaluación del modelo:** "
    }[nivel_usuario]
    return (
        f"{intro}Estoy leyendo los parámetros actuales de tu simulador:\n\n"
        f"> `{contexto_simulacion}`\n\n"
        f"Observa cómo responden las curvas al mover los controles. ¿Qué patrón notas ahora mismo en la gráfica?"
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
# INICIALIZACIÓN DE ESTADO DE SESIÓN
# ==============================================================================
if "telemetria" not in st.session_state:
    st.session_state.telemetria = "Pestaña inicial: Curvas de Neumáticos."
if "mision_previa" not in st.session_state:
    st.session_state.mision_previa = "Misión 1 (Neumáticos): ¿Por qué patinan las llantas?"
if "nivel_previo" not in st.session_state:
    st.session_state.nivel_previo = "🟢 Explorador (Sin ingeniería)"
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = [
        {"role": "assistant", "content": MISIONES_PEDAGOGICAS[st.session_state.mision_previa]}
    ]

# ==============================================================================
# BARRA LATERAL: TUTOR INTELIGENTE ADAPTATIVO (IA)
# ==============================================================================
with st.sidebar:
    st.header("🎓 Tutor Inteligente (IA)")
    nivel_usuario = st.radio(
        "1. Elige tu nivel de experiencia:",
        ["🟢 Explorador (Sin ingeniería)", "🟡 Estudiante (Fundamentos)", "🔴 Ingeniero (Análisis técnico)"],
        index=0
    )
    mision_actual = st.selectbox(
        "2. Misión Guiada (Aprende haciendo):",
        list(MISIONES_PEDAGOGICAS.keys()),
        index=1
    )

    if mision_actual != st.session_state.mision_previa or nivel_usuario != st.session_state.nivel_previo:
        st.session_state.mision_previa = mision_actual
        st.session_state.nivel_previo = nivel_usuario
        st.session_state.mensajes_chat = [
            {"role": "assistant", "content": MISIONES_PEDAGOGICAS[mision_actual]}
        ]

    st.divider()
    contenedor_chat = st.container(height=360)
    with contenedor_chat:
        for msg in st.session_state.mensajes_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    pregunta = st.chat_input("Escribe tu hipótesis, respuesta o duda...")
    if pregunta:
        st.session_state.mensajes_chat.append({"role": "user", "content": pregunta})
        with contenedor_chat:
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"):
                with st.spinner("Analizando tu simulación..."):
                    resp = consultar_gemini(
                        pregunta,
                        st.session_state.mensajes_chat[:-1],
                        nivel_usuario,
                        mision_actual,
                        st.session_state.telemetria
                    )
                    st.markdown(resp)
        st.session_state.mensajes_chat.append({"role": "assistant", "content": resp})

# ==============================================================================
# ÁREA PRINCIPAL: 5 PESTAÑAS INTERACTIVAS
# ==============================================================================
st.title("🏎️ Laboratorio Interactivo de Dinámica Vehicular")
st.caption("Experimenta cómo responde un automóvil al frenar, acelerar y tomar curvas. Abre la barra lateral izquierda (>) si deseas activar al Tutor IA.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Agarre de Llantas",
    "2. Manejo en Curva",
    "3. Dirección Ackermann",
    "4. Frenado ABS",
    "5. Fórmulas Matemáticas"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: NEUMÁTICOS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Comparativa de Modelos de Neumático (Pacejka vs. Julien vs. Dugoff)")
    st.write("Activa o desactiva cada modelo para observar cuánto agarre genera una llanta antes de comenzar a patinar.")

    if "t1_fz" not in st.session_state:
        st.session_state.t1_fz = 4000
        st.session_state.t1_mu = 0.90

    c_p1, c_p2, c_p3 = st.columns(3)
    if c_p1.button("☀️ Asfalto Seco (μ = 0.95)", use_container_width=True):
        st.session_state.t1_mu = 0.95
    if c_p2.button("🌧️ Pavimento Mojado (μ = 0.50)", use_container_width=True):
        st.session_state.t1_mu = 0.50
    if c_p3.button("❄️ Hielo / Nieve (μ = 0.20)", use_container_width=True):
        st.session_state.t1_mu = 0.20

    modelos_visibles = st.multiselect(
        "Modelos matemáticos a comparar en la gráfica:",
        ["Pacejka (Magic Formula)", "Julien (Teórico)", "Dugoff (Analítico)"],
        default=["Pacejka (Magic Formula)", "Julien (Teórico)", "Dugoff (Analítico)"]
    )

    col_s1, col_s2 = st.columns(2)
    Fz = col_s1.slider("Peso sobre la llanta (Carga Fz) [N]", 1500, 8000, key="t1_fz", step=250)
    mu = col_s2.slider("Agarre del suelo (μ): 0.2 Hielo | 0.5 Lluvia | 0.9 Seco", 0.15, 1.20, key="t1_mu", step=0.05)

    with st.expander("⚙️ Ajustes avanzados del neumático (Nivel Estudiante / Ingeniero)"):
        ca1, ca2 = st.columns(2)
        C_long_kN = ca1.slider("Rigidez al frenar/acelerar (C_κ) [kN]", 30, 150, 80, 5)
        C_lat_kN = ca2.slider("Dureza lateral en curva (C_α) [kN/rad]", 20, 120, 60, 5)
        cb1, cb2, cb3 = st.columns(3)
        B_pac = cb1.slider("Factor Rigidez Pacejka (B)", 4.0, 20.0, 10.0, 0.5)
        C_pac = cb2.slider("Factor Forma Pacejka (C)", 1.1, 2.2, 1.65, 0.05)
        E_pac = cb3.slider("Factor Curvatura Pacejka (E)", -1.0, 1.0, 0.2, 0.1)

    C_kappa = C_long_kN * 1000.0
    C_alpha = C_lat_kN * 1000.0
    D_max = mu * Fz
    kappa = np.linspace(0, 1.0, 300)
    alpha = np.linspace(0, 20.0, 300)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if "Pacejka (Magic Formula)" in modelos_visibles:
        ax1.plot(kappa * 100, calcular_pacejka_long(kappa, B_pac, C_pac, D_max, E_pac), label='Pacejka (Magic Formula)', color='#1f77b4', lw=2.4)
        ax2.plot(alpha, calcular_pacejka_lat(alpha, B_pac, C_pac, D_max, E_pac), label='Pacejka (Magic Formula)', color='#1f77b4', lw=2.4)
    if "Julien (Teórico)" in modelos_visibles:
        ax1.plot(kappa * 100, calcular_julien_long(kappa, Fz, mu, C_kappa), label='Julien (Teórico)', color='#ff7f0e', lw=2.1, ls='--')
        ax2.plot(alpha, calcular_julien_lat(alpha, Fz, mu, C_alpha), label='Julien (Teórico)', color='#ff7f0e', lw=2.1, ls='--')
    if "Dugoff (Analítico)" in modelos_visibles:
        ax1.plot(kappa * 100, calcular_dugoff_long(kappa, Fz, mu, C_kappa), label='Dugoff (Analítico)', color='#2ca02c', lw=2.1, ls='-.')
        ax2.plot(alpha, calcular_dugoff_lat(alpha, Fz, mu, C_alpha), label='Dugoff (Analítico)', color='#2ca02c', lw=2.1, ls='-.')

    ax1.axhline(D_max, color='gray', ls=':', lw=1, label=f'Límite Fricción ({D_max:.0f} N)')
    ax1.set_title('Fuerza Longitudinal (Acelerar / Frenar)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Deslizamiento Longitudinal $\\kappa$ [%]')
    ax1.set_ylabel('Fuerza $F_x$ [N]')
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.axhline(D_max, color='gray', ls=':', lw=1)
    ax2.set_title('Fuerza Lateral (Agarre en Curva)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Ángulo de Deriva $\\alpha$ [grados]')
    ax2.set_ylabel('Fuerza Lateral $F_y$ [N]')
    ax2.grid(True, ls='--', alpha=0.5)
    if modelos_visibles:
        ax2.legend(fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.session_state.telemetria = (
        f"[Pestaña 1 - Neumáticos] Modelos visibles: {modelos_visibles}. Carga Fz={Fz} N, "
        f"Agarre mu={mu}, Fuerza máxima={D_max:.0f} N."
    )

# ------------------------------------------------------------------------------
# PESTAÑA 2: MODELO BICICLETA
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Estabilidad Direccional: Subviraje vs. Sobreviraje (Modelo Bicicleta)")
    st.write("Desplaza el Centro de Gravedad (CG) y observa si el auto tiende a abrirse en curva (subviraje) o a hacer trompo (sobreviraje).")

    if "t2_a" not in st.session_state:
        st.session_state.t2_a = 1.10
        st.session_state.t2_b = 1.60
        st.session_state.t2_v = 80

    cb_1, cb_2, cb_3 = st.columns(3)
    if cb_1.button("🚗 Auto Familiar (Peso Adelante - Seguro)", use_container_width=True):
        st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.05, 1.65, 80
    if cb_2.button("🏎️ Deportivo Equilibrado (50/50)", use_container_width=True):
        st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.35, 1.35, 90
    if cb_3.button("⚠️ Auto Nervioso (Peso Atrás - Trompo)", use_container_width=True):
        st.session_state.t2_a, st.session_state.t2_b, st.session_state.t2_v = 1.70, 1.00, 95

    cs1, cs2, cs3 = st.columns(3)
    a = cs1.slider("Distancia CG al eje delantero (a) [m]", 0.70, 2.00, key="t2_a", step=0.05)
    b = cs2.slider("Distancia CG al eje trasero (b) [m]", 0.70, 2.00, key="t2_b", step=0.05)
    vel_kmh = cs3.slider("Velocidad de prueba [km/h]", 20, 160, key="t2_v", step=5)

    with st.expander("⚙️ Ajustes avanzados de masa y rigidez de viraje"):
        cm1, cm2, cm3 = st.columns(3)
        masa = cm1.slider("Masa total del auto [kg]", 800, 2500, 1500, 50)
        Caf_kN = cm2.slider("Agarre lateral delantero (C_αf) [kN/rad]", 30, 120, 65, 5)
        Car_kN = cm3.slider("Agarre lateral trasero (C_αr) [kN/rad]", 30, 120, 65, 5)

    L = a + b
    Wf = (masa * 9.81 * b) / L
    Wr = (masa * 9.81 * a) / L
    Kus = (Wf / (Caf_kN * 1000.0)) - (Wr / (Car_kN * 1000.0))
    Kus_deg_g = np.rad2deg(Kus * 9.81)

    ay_g = np.linspace(0, 0.85, 100)
    R_curva = 50.0
    delta_ack_deg = np.rad2deg(L / R_curva)
    delta_total_deg = delta_ack_deg + Kus_deg_g * ay_g

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([0, L], [0, 0], 'k-', lw=4)
    ax1.plot(0, 0, 'ks', markersize=14, label='Eje Delantero')
    ax1.plot(L, 0, 'ks', markersize=14, label='Eje Trasero')
    ax1.plot(a, 0, 'ro', markersize=16, label='Centro de Gravedad (CG)')
    ax1.annotate(f'Delante: {b/L*100:.0f}% peso', xy=(0, 0.15), ha='center', fontsize=9.5, fontweight='bold', color='#1f77b4')
    ax1.annotate(f'Atrás: {a/L*100:.0f}% peso', xy=(L, 0.15), ha='center', fontsize=9.5, fontweight='bold', color='#e74c3c')
    ax1.set_xlim(-0.5, L + 0.5)
    ax1.set_ylim(-0.5, 0.6)
    ax1.axis('off')
    ax1.set_title(f'Distribución de Masa (Batalla L = {L:.2f} m)', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower center', fontsize=8.5)

    color_curva = '#2980b9' if Kus > 0.0005 else ('#c0392b' if Kus < -0.0005 else '#27ae60')
    ax2.plot(ay_g, delta_total_deg, color=color_curva, lw=2.5, label='Ángulo de Volante Requerido')
    ax2.axhline(delta_ack_deg, color='gray', ls='--', label='Dirección Neutra (Ackermann)')
    ax2.set_title('Comportamiento en Curva (Skidpad R = 50 m)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Aceleración Lateral [g]')
    ax2.set_ylabel('Ángulo de Dirección en Ruedas [°]')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    if Kus_deg_g > 0.3:
        st.info(f"🔵 **SUBVIRAJE (Tendencia segura / abre trayectoria):** $K_{{us}} = +{Kus_deg_g:.2f}^\\circ/g$. Reparto: `{b/L*100:.0f}%` delante / `{a/L*100:.0f}%` atrás.")
    elif Kus_deg_g < -0.3:
        v_crit = np.sqrt((-L) / Kus) * 3.6
        st.error(f"🔴 **SOBREVIRAJE (Tendencia nerviosa / riesgo de trompo):** $K_{{us}} = {Kus_deg_g:.2f}^\\circ/g$. Velocidad crítica de inestabilidad: `{v_crit:.1f} km/h`.")
    else:
        st.success(f"🟢 **DIRECCIÓN NEUTRA (Equilibrio ideal):** $K_{{us}} = {Kus_deg_g:.2f}^\\circ/g$. Reparto `{b/L*100:.0f}% / {a/L*100:.0f}%`.")

    st.session_state.telemetria = (
        f"[Pestaña 2 - Modelo Bicicleta] Reparto={b/L*100:.0f}% delante / {a/L*100:.0f}% atrás, "
        f"Kus={Kus_deg_g:.2f} deg/g, Velocidad={vel_kmh} km/h."
    )

# ------------------------------------------------------------------------------
# PESTAÑA 3: DIRECCIÓN ACKERMANN
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Geometría de Dirección Ackermann")
    st.write("Al dar vuelta en una esquina, la rueda interior recorre un círculo más cerrado que la exterior y debe girar un ángulo mayor.")

    ca_1, ca_2 = st.columns(2)
    delta_in_deg = ca_1.slider("Giro del volante / Rueda Interna [°]", 0, 45, 25, 1)
    porcentaje_ack = ca_2.slider("Porcentaje Geometría Ackermann (0% Paralelo | 100% Ideal) [%]", 0, 120, 100, 10)

    ca_3, ca_4 = st.columns(2)
    L_ack = ca_3.slider("Largo entre ejes (Batalla L) [m]", 1.8, 3.5, 2.7, 0.1)
    W_ack = ca_4.slider("Ancho entre llantas (Trocha W) [m]", 1.2, 2.0, 1.6, 0.05)

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

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([-W_ack/2, W_ack/2, W_ack/2, -W_ack/2, -W_ack/2], [0, 0, L_ack, L_ack, 0], 'k-', lw=1.5, alpha=0.6)
    ax1.plot([-W_ack/2, -W_ack/2], [-0.25, 0.25], 'b-', lw=5)
    ax1.plot([W_ack/2, W_ack/2], [-0.25, 0.25], 'b-', lw=5)

    r_len = 0.35
    din_r = np.deg2rad(delta_in_deg)
    dout_r = np.deg2rad(delta_out_real_deg)
    ax1.plot([-W_ack/2 - r_len*np.sin(din_r), -W_ack/2 + r_len*np.sin(din_r)],
             [L_ack + r_len*np.cos(din_r), L_ack - r_len*np.cos(din_r)], 'r-', lw=5, label=f'Rueda Interna ({delta_in_deg:.1f}°)')
    ax1.plot([W_ack/2 - r_len*np.sin(dout_r), W_ack/2 + r_len*np.sin(dout_r)],
             [L_ack + r_len*np.cos(dout_r), L_ack - r_len*np.cos(dout_r)], 'g-', lw=5, label=f'Rueda Externa ({delta_out_real_deg:.1f}°)')
    ax1.set_xlim(-W_ack - 0.5, W_ack + 0.5)
    ax1.set_ylim(-0.6, L_ack + 0.8)
    ax1.set_aspect('equal')
    ax1.set_title('Vista Superior de Ruedas (Giro Izquierda)', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5)
    ax1.grid(True, ls=':', alpha=0.5)

    rango_in = np.linspace(0, 45, 100)
    rango_rad = np.deg2rad(rango_in)
    R_arr = np.where(rango_in > 0.1, L_ack / np.tan(np.maximum(rango_rad, 1e-4)) + W_ack / 2.0, 1e6)
    ideal_out = np.rad2deg(np.arctan(L_ack / (R_arr + W_ack / 2.0)))
    real_out = rango_in - factor * (rango_in - ideal_out)

    ax2.plot(rango_in, rango_in, 'k--', lw=1.2, label='Dirección Paralela (0% Ackermann)')
    ax2.plot(rango_in, ideal_out, 'b:', lw=1.8, label='Ackermann Puro (100%)')
    ax2.plot(rango_in, real_out, 'g-', lw=2.3, label=f'Configuración Actual ({porcentaje_ack}%)')
    ax2.plot(delta_in_deg, delta_out_real_deg, 'ro', markersize=8)
    ax2.set_xlabel('Ángulo Rueda Interna $\\delta_{in}$ [°]')
    ax2.set_ylabel('Ángulo Rueda Externa $\\delta_{out}$ [°]')
    ax2.set_title('Relación de Giro Interno vs. Externo', fontsize=11, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rueda Interna", f"{delta_in_deg:.1f}°")
    m2.metric("Rueda Externa", f"{delta_out_real_deg:.1f}°")
    m3.metric("Diferencia (Toe-out)", f"{diferencia_deg:.1f}°")
    m4.metric("Radio de Giro", f"{min(R_giro, 99.9):.1f} m")

    st.session_state.telemetria = (
        f"[Pestaña 3 - Ackermann] Ackermann={porcentaje_ack}%, Rueda interna={delta_in_deg:.1f} deg, "
        f"Rueda externa={delta_out_real_deg:.1f} deg."
    )

# ------------------------------------------------------------------------------
# PESTAÑA 4: SIMULADOR DE FRENADO ABS
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
    st.subheader("Simulador de Frenado de Emergencia (Sistema ABS)")
    st.write("Compara el frenado con ABS encendido vs. apagado y contrasta cómo predice cada modelo matemático la distancia de detención.")

    if "t4_v" not in st.session_state:
        st.session_state.t4_v = 100
        st.session_state.t4_mu = 0.85
        st.session_state.t4_tq = 2400

    b_abs1, b_abs2, b_abs3 = st.columns(3)
    if b_abs1.button("☀️ Asfalto Seco (100 km/h)", use_container_width=True):
        st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 100, 0.85, 2400
    if b_abs2.button("🌧️ Pavimento Mojado (100 km/h)", use_container_width=True):
        st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 100, 0.45, 2400
    if b_abs3.button("❄️ Hielo Negro (80 km/h)", use_container_width=True):
        st.session_state.t4_v, st.session_state.t4_mu, st.session_state.t4_tq = 80, 0.20, 2400

    c_ctrl1, c_ctrl2 = st.columns(2)
    abs_activo = c_ctrl1.toggle("🛡️ Sistema ABS Encendido (Evita bloqueo de llanta)", value=True)
    modelo_abs = c_ctrl2.selectbox("Modelo físico de llanta:", ["Comparar los 3 Modelos", "Pacejka", "Julien", "Dugoff"], index=0)

    c_sl1, c_sl2, c_sl3 = st.columns(3)
    v_ini_kmh = c_sl1.slider("Velocidad inicial [km/h]", 30, 160, key="t4_v", step=5)
    mu_abs = c_sl2.slider("Agarre del suelo (μ)", 0.15, 1.10, key="t4_mu", step=0.05)
    torque_max = c_sl3.slider("Fuerza aplicada al pedal de freno [N·m]", 500, 3500, key="t4_tq", step=100)

    estado_txt = "ENCENDIDO" if abs_activo else "APAGADO"
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)

    if modelo_abs == "Comparar los 3 Modelos":
        estilos = [("Pacejka", "#1f77b4", "-"), ("Julien", "#ff7f0e", "--"), ("Dugoff", "#2ca02c", "-.")]
        resumen_lista = []
        cols_res = st.columns(3)
        for idx, (nombre, color, ls) in enumerate(estilos):
            ht, hV, hVw, hk, dist, t_det, t_bloq, Fz = _correr_integracion_abs(
                nombre, abs_activo, v_ini_kmh, mu_abs, torque_max
            )
            ax1.plot(ht, hV, label=f'Vel. Auto ({nombre}: {dist:.1f} m)', color=color, lw=2.3, ls=ls)
            ax2.plot(ht, hk, label=f'Patinaje κ ({nombre})', color=color, lw=2.0, ls=ls)
            cols_res[idx].metric(f"Modelo {nombre}", f"{dist:.2f} m", f"Tiempo: {t_det:.2f} s | Bloq: {t_bloq:.2f} s")
            resumen_lista.append(f"{nombre}: {dist:.2f}m ({t_det:.2f}s)")

        ax1.set_ylabel('Velocidad Auto [km/h]')
        ax1.set_title(f'Comparativa de Frenado (Pacejka vs Julien vs Dugoff) | ABS {estado_txt}', fontsize=11, fontweight='bold')
        ax1.grid(True, ls='--', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=8.5)

        ax2.axhspan(12, 18, color='#f1c40f', alpha=0.25, label='Zona Óptima ABS (12% - 18%)')
        ax2.axhline(100, color='#c0392b', ls=':', lw=1.2, label='Bloqueo Total (100%)')
        ax2.set_xlabel('Tiempo de Frenado [s]')
        ax2.set_ylabel('Deslizamiento $\\kappa$ [%]')
        ax2.set_ylim(-5, 110)
        ax2.grid(True, ls='--', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        st.session_state.telemetria = (
            f"[Pestaña 4 - Simulador ABS Comparativo] ABS={estado_txt}, V={v_ini_kmh} km/h, mu={mu_abs}, "
            f"Resultados: {', '.join(resumen_lista)}"
        )
    else:
        ht, hV, hVw, hk, dist, t_det, t_bloq, Fz = _correr_integracion_abs(
            modelo_abs, abs_activo, v_ini_kmh, mu_abs, torque_max
        )
        ax1.plot(ht, hV, label='Velocidad del Auto ($V$)', color='#1f77b4', lw=2.5)
        ax1.plot(ht, hVw, label='Velocidad de Giro de la Llanta ($\\omega R_w$)', color='#e74c3c', lw=2.0, ls='--')
        ax1.set_ylabel('Velocidad [km/h]')
        ax1.set_title(f'Prueba de Frenado | Modelo: {modelo_abs} | ABS {estado_txt}', fontsize=11, fontweight='bold')
        ax1.grid(True, ls='--', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=9)

        ax2.plot(ht, hk, label='Deslizamiento ($\\kappa$)', color='#27ae60', lw=2.2)
        ax2.axhspan(12, 18, color='#f1c40f', alpha=0.25, label='Zona Óptima ABS (12% - 18%)')
        ax2.axhline(100, color='#c0392b', ls=':', lw=1.2, label='Bloqueo Total (100%)')
        ax2.set_xlabel('Tiempo de Frenado [s]')
        ax2.set_ylabel('Deslizamiento $\\kappa$ [%]')
        ax2.set_ylim(-5, 110)
        ax2.grid(True, ls='--', alpha=0.6)
        ax2.legend(loc='upper right', fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        r1, r2, r3 = st.columns(3)
        r1.metric("Distancia de Frenado", f"{dist:.2f} m")
        r2.metric("Tiempo Total", f"{t_det:.2f} s")
        r3.metric("Tiempo Llanta Bloqueada", f"{t_bloq:.2f} s")

        st.session_state.telemetria = (
            f"[Pestaña 4 - Simulador ABS] ABS={estado_txt}, Modelo={modelo_abs}, V={v_ini_kmh} km/h, "
            f"mu={mu_abs}, Distancia={dist:.2f} m, Tiempo={t_det:.2f} s, Bloqueo={t_bloq:.2f} s."
        )

# ------------------------------------------------------------------------------
# PESTAÑA 5: FORMULARIO MATEMÁTICO DE REFERENCIA
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("📐 Formulario Matemático del Simulador de Dinámica Vehicular")
    st.markdown(r"""
    Aquí puedes consultar las ecuaciones exactas que gobiernan cada módulo del laboratorio para conectar las gráficas con su fundamento analítico.

    ---

    ### 1. Modelos Matemáticos del Neumático (Pestaña 1)

    #### A. Variables de Deslizamiento
    * **Deslizamiento longitudinal ($\kappa$):**
      $$\kappa = \frac{V - \omega R_w}{\max(V, \, \omega R_w)}$$
    * **Ángulo de deriva lateral ($\alpha$):** Ángulo entre la orientación geométrica de la rueda y su vector de velocidad real en el plano de contacto.

    #### B. Fórmula Mágica de Pacejka (*Magic Formula*)
    $$F(x) = D \sin\Big( C \arctan\big( B x - E [B x - \arctan(B x)] \big) \Big)$$
    * Donde $x$ representa $\kappa$ (longitudinal) o $\alpha$ (lateral en radianes), $D = \mu F_z$ es el pico de adherencia, $B$ el factor de rigidez, $C$ el factor de forma y $E$ el factor de curvatura.

    #### C. Modelo Teórico de Julien
    $$\lambda_J = \frac{\mu F_z}{2 C_\kappa |\kappa|} \quad \text{(longitudinal)} \qquad \lambda_J = \frac{\mu F_z}{2 C_\alpha |\tan \alpha|} \quad \text{(lateral)}$$
    $$F_x = \begin{cases} C_\kappa \kappa & \text{si } \lambda_J \ge 1 \text{ (adherencia total)} \\ \mu F_z \left(1 - \dfrac{\mu F_z}{4 C_\kappa |\kappa|}\right)\text{sgn}(\kappa) & \text{si } \lambda_J < 1 \text{ (deslizamiento parcial)} \end{cases}$$

    #### D. Modelo Analítico de Dugoff
    $$\lambda = \frac{\mu F_z (1 - |\kappa|)}{2 C_\kappa |\kappa|} \quad \text{(longitudinal)} \qquad \lambda = \frac{\mu F_z}{2 C_\alpha |\tan \alpha|} \quad \text{(lateral)}$$
    $$f(\lambda) = \begin{cases} (2 - \lambda)\lambda & \text{si } \lambda < 1 \\ 1 & \text{si } \lambda \ge 1 \end{cases}$$
    $$F_x = \frac{C_\kappa \kappa}{1 - |\kappa|} f(\lambda) \qquad \qquad F_y = C_\alpha \tan(\alpha) f(\lambda)$$

    ---

    ### 2. Modelo Bicicleta de 2 Grados de Libertad (Pestaña 2)
    $$W_f = \frac{m g \, b}{L} \qquad \qquad W_r = \frac{m g \, a}{L} \qquad \qquad L = a + b$$
    $$K_{us} = \frac{W_f}{C_{\alpha f}} - \frac{W_r}{C_{\alpha r}} \quad [\text{rad}/(\text{m/s}^2)] \qquad \qquad \delta = \frac{L}{R} + K_{us} \, a_y$$
    * **$K_{us} > 0$ (Subviraje):** El eje delantero satura antes; se requiere mayor ángulo de volante al aumentar la aceleración lateral $a_y$.
    * **$K_{us} < 0$ (Sobreviraje):** El eje trasero pierde adherencia primero; velocidad crítica de inestabilidad: $V_{\text{crítica}} = \sqrt{-L / K_{us}}$.

    ---

    ### 3. Geometría de Dirección Ackermann (Pestaña 3)
    $$\cot(\delta_{\text{out, ideal}}) - \cot(\delta_{\text{in}}) = \frac{W}{L} \qquad \qquad \delta_{\text{out, real}} = \delta_{\text{in}} - p \, (\delta_{\text{in}} - \delta_{\text{out, ideal}})$$

    ---

    ### 4. Dinámica Longitudinal de Frenado y Control ABS (Pestaña 4)
    $$m_{\text{cuarto}} \frac{dV}{dt} = -F_x(\kappa) \qquad \qquad I_w \frac{d\omega}{dt} = F_x(\kappa) R_w - T_{\text{freno}}(t)$$
    $$T_{\text{freno}}(t) = \begin{cases} 0.10 \, T_{\text{máx}} & \text{si } \kappa > 0.18 \text{ (válvula de alivio activa)} \\ T_{\text{máx}} & \text{si } \kappa < 0.12 \text{ (reaplicación de presión)} \end{cases}$$
    """)
