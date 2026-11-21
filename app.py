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
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
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
