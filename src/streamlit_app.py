# src/streamlit_app.py — Predictor de Mercados
# Corre en Hugging Face Spaces con Streamlit
# Combina precios reales de yfinance + NLP con FinBERT + modelo LSTM propio

import streamlit as st          # framework de la interfaz web
import torch                    # PyTorch para cargar y correr el modelo
import torch.nn as nn           # para definir la arquitectura del modelo
import numpy as np              # operaciones numéricas
import pandas as pd             # manejo de datos tabulares
import plotly.graph_objects as go  # gráficas interactivas
import yfinance as yf           # precios históricos en tiempo real
import requests                 # llamadas HTTP a NYT y Giphy
import joblib                   # para cargar el scaler
import json                     # para cargar hiperparámetros y features
import os                       # para leer secrets del entorno
import random                   # para elegir GIF al azar
import time                     # para pausas entre reintentos
import pytz                     # para conversión de zona horaria
from io import StringIO         # para leer CSV desde string
from datetime import datetime, timedelta, timezone  # manejo de fechas
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # FinBERT

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ─────────────────────────────────────────────

# Esto debe ser la primera llamada de Streamlit — configura título, ícono y layout
st.set_page_config(
    page_title = "Predictor de Mercados",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# CSS personalizado — fondo claro con bordes remarcados y tipografía legible
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }

    /* Tarjetas de sectores — fondo blanco con borde sólido y acento de color */
    .sector-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 2px solid #e0e0e0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .sector-card-down {
        border-left: 5px solid #f44336;
    }

    /* Caja de predicción — fondo blanco con borde remarcado */
    .prediction-box {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 16px 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Divisor horizontal suave */
    .custom-divider {
        border: none;
        height: 1px;
        background: #e0e0e0;
        margin: 20px 0;
    }

    /* Caja de información — fondo azul muy claro con borde izquierdo azul */
    .info-box {
        background: #f0f4ff;
        border: 1px solid #c5d3f0;
        border-left: 4px solid #2196F3;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88em;
        color: #333333;
        line-height: 1.6;
        margin: 12px 0;
    }

    /* Caja de advertencia — fondo amarillo muy claro */
    .warning-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-left: 4px solid #f4a261;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88em;
        color: #7a5c00;
        line-height: 1.6;
        margin: 12px 0;
    }

    /* Caja de error — fondo rojo muy claro */
    .error-box {
        background: #fff0f0;
        border: 1px solid #ffcccc;
        border-left: 4px solid #e63946;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88em;
        color: #8b0000;
        line-height: 1.6;
        margin: 12px 0;
    }

    /* Tarjetas técnicas del lado derecho en "Sobre el Proyecto" */
    .tech-card {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .tech-card-title {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .tech-card-label {
        font-size: 0.72em;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .tech-card-desc {
        font-size: 0.82em;
        color: #555555;
        line-height: 1.6;
    }

    /* Ocultamos elementos de Streamlit que no necesitamos */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Estilos de las pestañas de navegación */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 1em; font-weight: 500; }

    /* Indicadores de estado del mercado */
    .market-open  { color: #2e7d32; font-weight: 600; font-size: 0.85em; }
    .market-close { color: #c62828; font-weight: 600; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES Y CONFIGURACIÓN
# ─────────────────────────────────────────────

# Key de NYT
   NYT_KEY = os.environ.get("NYT_API_KEY", "")

# Key de Giphy desde los secrets de Hugging Face
# os.environ lee las variables de entorno donde HF guarda los secrets
GIPHY_KEY   = os.environ.get("GIPHY_API_KEY", "")

# Zona horaria de México — el servidor de HF corre en UTC
ZONA_MEXICO = pytz.timezone("America/Monterrey")

# Los 10 sectores con metadata completa
SECTORES = {
    "XLK":  {"nombre": "Tecnología",           "empresas": "Apple, Microsoft, Nvidia"},
    "XLF":  {"nombre": "Finanzas",             "empresas": "JPMorgan, Berkshire, BofA"},
    "XLE":  {"nombre": "Energía",              "empresas": "ExxonMobil, Chevron"},
    "XLI":  {"nombre": "Industriales",         "empresas": "Boeing, Caterpillar, GE"},
    "XLC":  {"nombre": "Comunicaciones",       "empresas": "Meta, Alphabet, Netflix"},
    "XLY":  {"nombre": "Consumo discrecional", "empresas": "Amazon, Tesla, McDonald's"},
    "XLP":  {"nombre": "Consumo básico",       "empresas": "P&G, Walmart, Coca-Cola"},
    "XLU":  {"nombre": "Utilities",            "empresas": "NextEra, Duke Energy"},
    "XLRE": {"nombre": "Bienes raíces",        "empresas": "Simon Property, AvalonBay"},
    "XLB":  {"nombre": "Materiales",           "empresas": "Linde, Freeport-McMoRan"},
}

# Frases de recomendación según predicción y nivel de confianza
RECOMENDACIONES = {
    ("sube", "alta"): {
        "frase":  "Los algoritmos están eufóricos. Las señales apuntan hacia arriba con fuerza.",
        "accion": "El modelo predice subida con alta confianza. Momento favorable para posiciones largas.",
        "color":  "#4CAF50"
    },
    ("sube", "media"): {
        "frase":  "El modelo dice que sube... pero como que no está muy seguro.",
        "accion": "Señal positiva moderada. Considera una posición pequeña con stop-loss ajustado.",
        "color":  "#8BC34A"
    },
    ("baja", "alta"): {
        "frase":  "El oso llegó a la fiesta y se comió todo el buffet.",
        "accion": "El modelo predice baja con alta confianza. Considera reducir exposición.",
        "color":  "#f44336"
    },
    ("baja", "media"): {
        "frase":  "El modelo dice que baja pero tampoco apostemos el rancho.",
        "accion": "Señal negativa moderada. Precaución recomendada, espera confirmación.",
        "color":  "#FF5722"
    },
    ("neutral", "baja"): {
        "frase":  "Ni yo sé qué va a pasar. El mercado tampoco.",
        "accion": "Señal ambigua. No hay suficiente confianza para recomendar una acción.",
        "color":  "#9E9E9E"
    },
}

# Búsquedas en Giphy según el resultado — todas relacionadas a stocks
GIPHY_QUERIES = {
    ("sube", "alta"):    "stocks up wall street celebration",
    ("sube", "media"):   "stock market rising bull",
    ("baja", "alta"):    "stock market crash bear",
    ("baja", "media"):   "stocks down trading sad",
    ("neutral", "baja"): "confused stocks trading",
}

# Queries NYT por sector para las anotaciones de la gráfica histórica
QUERIES_NYT = {
    "XLK":  "technology stocks market Wall Street",
    "XLF":  "bank financial stocks Wall Street market",
    "XLE":  "energy oil stocks market prices",
    "XLI":  "industrial manufacturing stocks market",
    "XLC":  "media technology stocks market revenue",
    "XLY":  "consumer retail stocks market spending",
    "XLP":  "consumer prices inflation stocks market",
    "XLU":  "energy utility stocks market electricity",
    "XLRE": "real estate housing stocks market",
    "XLB":  "commodities materials stocks market metals",
}

# ─────────────────────────────────────────────
# FUNCIONES DE CARGA DE MODELOS
# ─────────────────────────────────────────────

# @st.cache_resource le dice a Streamlit que solo cargue esto UNA vez
# Sin esto, cada vez que el usuario interactúa con la app
# volvería a descargar FinBERT (438 MB) — haría la app inutilizable
@st.cache_resource
def cargar_finbert():
    """Carga FinBERT desde Hugging Face Hub — solo se ejecuta una vez."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

@st.cache_resource
def cargar_modelo_lstm():
    """
    Carga el modelo LSTM entrenado, el scaler y la lista de features.
    Usa rutas absolutas basadas en la ubicación del script para garantizar
    que funciona independientemente del directorio de trabajo del servidor.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Definimos la arquitectura — debe ser idéntica a como la entrenamos
    # Si cambia un solo parámetro, los pesos no cargan correctamente
    class LSTMConAttention(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, dropout):
            super(LSTMConAttention, self).__init__()
            self.lstm = nn.LSTM(
                input_size    = input_dim,
                hidden_size   = hidden_dim,
                num_layers    = n_layers,
                dropout       = dropout if n_layers > 1 else 0,
                batch_first   = True,
                bidirectional = True
            )
            self.attention    = nn.Linear(hidden_dim * 2, 1)
            self.clasificador = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2)
            )

        def forward(self, x):
            lstm_out, _       = self.lstm(x)
            attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context           = (lstm_out * attention_weights).sum(dim=1)
            return self.clasificador(context)

    # Cargamos los hiperparámetros que guardamos durante el entrenamiento
    with open(os.path.join(BASE_DIR, "hiperparametros.json"), "r") as f:
        hp = json.load(f)

    # Instanciamos el modelo con los mismos hiperparámetros del entrenamiento
    modelo = LSTMConAttention(
        input_dim  = hp["input_dim"],
        hidden_dim = hp["hidden_dim"],
        n_layers   = hp["n_layers"],
        dropout    = hp["dropout"]
    )

    # Cargamos los pesos entrenados
    # map_location="cpu" porque el Space no tiene GPU
    modelo.load_state_dict(torch.load(
        os.path.join(BASE_DIR, "modelo_lstm.pt"), map_location="cpu"
    ))
    modelo.eval()  # modo inferencia — desactiva dropout

    # Cargamos el scaler y el orden de features
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "features.json"), "r") as f:
        features = json.load(f)

    return modelo, scaler, features, hp["lookback"]

# ─────────────────────────────────────────────
# FUNCIONES DE NEGOCIO
# ─────────────────────────────────────────────

def hora_mexico():
    """Regresa la hora actual en zona horaria de México (Monterrey)."""
    return datetime.now(timezone.utc).astimezone(ZONA_MEXICO)

def mercado_abierto():
    """
    Verifica si el mercado americano está abierto ahora mismo.
    Opera de lunes a viernes de 9:30 AM a 4:00 PM hora de Nueva York.
    """
    ny    = pytz.timezone("America/New_York")
    ahora = datetime.now(timezone.utc).astimezone(ny)
    if ahora.weekday() >= 5:
        return False
    apertura = ahora.replace(hour=9,  minute=30, second=0, microsecond=0)
    cierre   = ahora.replace(hour=16, minute=0,  second=0, microsecond=0)
    return apertura <= ahora <= cierre

def descargar_yahoo(simbolo, periodo="2y"):
    """
    Descarga precios usando yfinance que internamente usa curl_cffi
    para evitar los bloqueos de Yahoo Finance desde servidores en la nube.
    """
    try:
        ticker = yf.Ticker(simbolo)
        df = ticker.history(period=periodo, interval="1d", auto_adjust=True)

        if df.empty or len(df) < 5:
            return None

        # Renombramos columnas a minúscula para consistencia en el resto del código
        df = df.rename(columns={
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
            "Volume": "volume"
        })

        # Conservamos solo las columnas OHLCV — descartamos Dividends y Stock Splits
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df   = df[cols]

        # Normalizamos el índice a fecha sin timezone
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df       = df.dropna(subset=["close"])

        return df if len(df) >= 5 else None

    except Exception:
        return None

def obtener_precios_recientes(simbolo, lookback=30):
    """
    Descarga los últimos 60 días de precios para la predicción.
    Si falla con 60 días intenta períodos más largos para garantizar
    tener al menos lookback + 21 días de historial (necesario para return_21d).
    """
    for periodo in ["60d", "3mo", "6mo"]:
        df = descargar_yahoo(simbolo, periodo=periodo)
        if df is not None and len(df) >= lookback + 21:
            return df
    return None

def obtener_datos_historicos_completos(simbolo):
    """
    Descarga 2 años de precios para la gráfica y la tabla histórica.
    Si falla con 2 años intenta períodos más cortos.
    """
    for periodo in ["2y", "1y", "6mo"]:
        df = descargar_yahoo(simbolo, periodo=periodo)
        if df is not None and not df.empty:
            return df
    return None

def calcular_sentimiento_finbert(texto, tokenizer, model):
    """
    Calcula sentimiento financiero de un texto usando FinBERT.
    Acepta desde un titular hasta un artículo completo (máximo 512 tokens).
    Si el texto excede 512 tokens FinBERT lo trunca automáticamente.
    """
    if not texto or len(texto.strip()) == 0:
        return {"positivo": 0.0, "negativo": 0.0, "neutro": 1.0, "score_neto": 0.0}

    inputs = tokenizer(
        texto,
        return_tensors = "pt",
        truncation     = True,   # trunca si excede 512 tokens
        max_length     = 512,
        padding        = True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
    return {
        "positivo":   float(probs[0]),
        "negativo":   float(probs[1]),
        "neutro":     float(probs[2]),
        "score_neto": float(probs[0] - probs[1])
    }

def construir_features_para_prediccion(df, sentimiento, scaler, features):
    """
    Construye el vector de features replicando exactamente el preprocesamiento
    del entrenamiento para garantizar compatibilidad con el modelo.
    """
    df = df.copy().sort_index()

    # Features financieras — mismas fórmulas que en el notebook de entrenamiento
    df["return_1d"]        = df["close"].pct_change(1)
    df["return_5d"]        = df["close"].pct_change(5)
    df["return_21d"]       = df["close"].pct_change(21)
    df["volatilidad_10d"]  = df["return_1d"].rolling(10).std()
    df["volumen_relativo"] = df["volume"] / df["volume"].rolling(20).mean()
    df["return_lag1"]      = df["return_1d"].shift(1)
    df["return_lag2"]      = df["return_1d"].shift(2)
    df["return_lag3"]      = df["return_1d"].shift(3)
    df["rango_diario"]     = (df["high"] - df["low"]) / df["close"]

    # Features de sentimiento — las mismas 5 columnas del entrenamiento
    df["sent_positivo"]     = sentimiento["positivo"]
    df["sent_negativo"]     = sentimiento["negativo"]
    df["sent_neutro"]       = sentimiento["neutro"]
    df["sent_score_neto"]   = sentimiento["score_neto"]
    df["sent_num_noticias"] = 1  # el usuario proporcionó texto

    df = df.dropna(subset=features)
    if len(df) == 0:
        return None

    # Normalizamos con el scaler entrenado — transform, no fit_transform
    # fit_transform aquí causaría data leakage
    X = scaler.transform(df[features])
    return X

def predecir(simbolo, texto, modelo, tokenizer, finbert, scaler, features, lookback):
    """
    Pipeline completo de predicción:
    1. Descarga precios recientes
    2. Calcula sentimiento del texto con FinBERT
    3. Construye features combinando precios y sentimiento
    4. Crea secuencia temporal de lookback días
    5. Corre el modelo LSTM
    6. Regresa predicción, probabilidad y sentimiento
    """
    df = obtener_precios_recientes(simbolo, lookback)
    if df is None or len(df) < lookback + 21:
        return None, None, None, "datos"

    sentimiento = calcular_sentimiento_finbert(texto, tokenizer, finbert)

    X = construir_features_para_prediccion(df, sentimiento, scaler, features)
    if X is None or len(X) < lookback:
        return None, None, None, "features"

    # Tomamos los últimos "lookback" días como ventana de predicción
    secuencia = X[-lookback:]
    tensor    = torch.FloatTensor(secuencia).unsqueeze(0)  # añadimos dimensión batch

    with torch.no_grad():
        output       = modelo(tensor)
        probabilidad = torch.softmax(output, dim=1)[0].numpy()

    clase     = int(probabilidad.argmax())  # 0=baja, 1=sube
    confianza = float(probabilidad[clase])

    return clase, confianza, sentimiento, None

def obtener_gif_giphy(query):
    """
    Busca GIFs en Giphy y elige uno al azar entre los primeros 10 resultados.
    Si falla regresa None y la app continúa sin GIF.
    """
    if not GIPHY_KEY:
        return None
    try:
        r    = requests.get(
            "https://api.giphy.com/v1/gifs/search",
            params={"api_key": GIPHY_KEY, "q": query, "limit": 10, "rating": "g"},
            timeout=5
        )
        data = r.json().get("data", [])
        if data:
            return random.choice(data)["images"]["original"]["url"]
    except Exception:
        pass
    return None

def obtener_noticias_nyt(query, dias=30):
    """
    Descarga noticias recientes del NYT para anotar la gráfica histórica.
    Solo se usa para las anotaciones visuales — no afecta la predicción.
    """
    fecha_fin = datetime.now()
    fecha_ini = fecha_fin - timedelta(days=dias)
    try:
        r    = requests.get(
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            params={
                "q":          query,
                "begin_date": fecha_ini.strftime("%Y%m%d"),
                "end_date":   fecha_fin.strftime("%Y%m%d"),
                "sort":       "relevance",
                "api-key":    NYT_KEY,
            },
            timeout=10
        )
        docs = r.json().get("response", {}).get("docs", [])
        return [{"fecha": d["pub_date"][:10], "titular": d["headline"]["main"]}
                for d in docs[:5]]
    except Exception:
        return []

def construir_tabla_historica(df):
    """
    Construye una tabla profesional tipo Wall Street con métricas clave
    a partir del DataFrame de precios históricos.
    """
    tabla = pd.DataFrame()
    tabla["Fecha"]          = df.index.strftime("%d %b %Y")
    tabla["Apertura"]       = df["open"].round(2)
    tabla["Máximo"]         = df["high"].round(2)
    tabla["Mínimo"]         = df["low"].round(2)
    tabla["Cierre"]         = df["close"].round(2)

    # Variación porcentual respecto al día anterior
    tabla["Variación %"]    = df["close"].pct_change(1).mul(100).round(2)

    # Volumen en millones para facilitar la lectura
    tabla["Volumen (M)"]    = (df["volume"] / 1_000_000).round(2)

    # Volumen relativo — necesita 20 días de historial, los primeros días quedan NaN
    vol_promedio            = df["volume"].rolling(20).mean()
    tabla["Vol. Relativo"]  = (df["volume"] / vol_promedio).round(2)

    # Rango diario — amplitud de la vela como porcentaje del cierre
    tabla["Rango Diario %"] = ((df["high"] - df["low"]) / df["close"]).mul(100).round(2)

    # Ordenamos de más reciente a más antiguo para ver primero lo actual
    tabla = tabla.iloc[::-1].reset_index(drop=True)

    # Solo eliminamos filas donde el Cierre sea NaN — dato esencial
    # Las demás columnas pueden tener NaN sin problema
    return tabla.dropna(subset=["Cierre"])

# ─────────────────────────────────────────────
# PANTALLA DE BIENVENIDA
# ─────────────────────────────────────────────

def mostrar_bienvenida(gif_url=None):
    """
    Pantalla inicial con GIF pequeño de Giphy, título y botón de entrada.
    El GIF se pasa como parámetro desde main() para evitar llamadas
    antes de que Streamlit inicialice la sesión correctamente.
    """
    col_izq, col_centro, col_der = st.columns([1, 3, 1])
    with col_centro:

        # GIF pequeño centrado — columnas internas para controlar el tamaño
        if gif_url:
            g1, g2, g3 = st.columns([1, 2, 1])
            with g2:
                st.image(gif_url, use_container_width=True)

        st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <h1 style='font-size:2.8em; font-weight:bold;
                background: linear-gradient(90deg, #4CAF50, #2196F3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;'>
                Predictor de Mercados
            </h1>
            <p style='font-size:1em; color:#555555; margin-top:6px;'>
                LSTM + Transformers aplicados a los mercados financieros
            </p>
            <p style='color:#777777; font-size:0.88em; margin-top:4px;'>
                Eric Brandon García Luján
            </p>
            <p style='color:#aaaaaa; font-size:0.8em; margin-top:2px;'>
                UANL · Maestría en Ciencia de Datos · 2026
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Modelo LSTM")
            # Tiempo verbal formal — "Se entrenó" en lugar de "entrenamos"
            st.write(
                "Se entrenó una red neuronal con 2 años de datos reales del S&P 500 "
                "para detectar patrones en el comportamiento histórico del precio."
            )
        with c2:
            st.markdown("### Análisis de noticias")
            # Descripción enfocada en el objetivo real del componente NLP
            st.write(
                "Se intenta predecir, a través de titulares y artículos financieros, "
                "si el mercado podrá subir o bajar acorde a esa noticia."
            )
        with c3:
            st.markdown("### 10 Sectores")
            # "Este trabajo cubre" en lugar de "cubrimos"
            st.write(
                "Este trabajo cubre las grandes industrias del mercado americano: "
                "tecnología, energía, finanzas y más."
            )

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # Botón cambiado de "Explorar el mercado" a "Ir al modelo"
        if st.button("Ir al modelo", use_container_width=True):
            st.session_state["bienvenida_vista"] = True
            st.rerun()

# ─────────────────────────────────────────────
# SECCIÓN: DASHBOARD GENERAL
# ─────────────────────────────────────────────

def mostrar_dashboard(modelo, scaler, features, lookback, tokenizer, finbert):
    """
    Vista general de todos los sectores con precios en tiempo real.
    Incluye indicador de estado del mercado y botón de actualización.
    """
    # Header con estado del mercado y botón de actualizar
    col_titulo, col_estado, col_btn = st.columns([3, 2, 1])
    with col_titulo:
        st.markdown("## Vista general del mercado")
        st.caption(
            f"Última actualización: {hora_mexico().strftime('%d %b %Y %H:%M')} (hora Monterrey)"
        )
    with col_estado:
        abierto = mercado_abierto()
        estado  = "Mercado abierto" if abierto else "Mercado cerrado"
        clase   = "market-open" if abierto else "market-close"
        st.markdown(
            f"<div style='padding-top:20px'>"
            f"<span class='{clase}'>● {estado}</span>"
            f"<div style='font-size:0.72em; color:#888888; margin-top:2px'>"
            f"{'Precios en tiempo real' if abierto else 'Mostrando último cierre'}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col_btn:
        # Botón para forzar actualización de datos limpiando el cache
        if st.button("Actualizar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Descripción actualizada — "En esta sección se puede observar" en lugar de "Aquí puedes ver"
    st.markdown("""
    <div class='info-box'>
        En esta sección se puede observar los 10 sectores utilizados en este proyecto
        y cada tarjeta muestra el precio actual y el cambio porcentual respecto al día anterior.
        El mercado americano opera de lunes a viernes de 8:30 AM a 3:00 PM hora de Monterrey.
        Fuera de ese horario se muestra el último precio de cierre disponible.
    </div>
    """, unsafe_allow_html=True)

    # Descargamos cada sector individualmente con reintentos por período
    # yfinance desde servidores de HF en Europa falla frecuentemente
    # por bloqueos de Yahoo Finance — intentar múltiples períodos garantiza datos
    with st.spinner("Descargando datos del mercado..."):
        datos     = {}
        tipo_dato = {}
        for simbolo in SECTORES.keys():
            for periodo in ["5d", "60d", "3mo"]:
                df_tmp = descargar_yahoo(simbolo, periodo=periodo)
                if df_tmp is not None and len(df_tmp) >= 2:
                    datos[simbolo]     = df_tmp["close"]
                    tipo_dato[simbolo] = "En vivo" if periodo == "5d" else "Último cierre"
                    break

    if len(datos) == 0:
        st.markdown("""
        <div class='warning-box'>
            No se pudieron obtener datos del mercado en este momento. Yahoo Finance está
            bloqueando temporalmente las solicitudes desde el servidor. No es un error del modelo.
            Haz clic en "Actualizar" o recarga la página en unos segundos.
        </div>
        """, unsafe_allow_html=True)
        return

    if len(datos) < len(SECTORES):
        st.caption(
            f"Se obtuvieron datos de {len(datos)} de {len(SECTORES)} sectores. "
            "Algunos pueden aparecer sin datos temporalmente."
        )

    # Mostramos las tarjetas en filas de 5 sectores
    cols = st.columns(5)
    for i, (simbolo, info) in enumerate(SECTORES.items()):
        with cols[i % 5]:
            try:
                precio_hoy  = float(datos[simbolo].iloc[-1])
                precio_ayer = float(datos[simbolo].iloc[-2])
                cambio      = ((precio_hoy - precio_ayer) / precio_ayer) * 100
                color       = "#2e7d32" if cambio >= 0 else "#c62828"
                flecha      = "▲" if cambio >= 0 else "▼"
                fuente      = tipo_dato.get(simbolo, "")

                st.markdown(f"""
                <div class='sector-card {"" if cambio >= 0 else "sector-card-down"}'>
                    <div style='font-weight:bold; font-size:0.95em; color:#222222'>{info['nombre']}</div>
                    <div style='color:#888888; font-size:0.72em; margin-bottom:8px'>{simbolo}</div>
                    <div style='color:{color}; font-size:1.4em; font-weight:bold'>
                        {flecha} {abs(cambio):.2f}%
                    </div>
                    <div style='color:#555555; font-size:0.88em'>${precio_hoy:.2f}</div>
                    <div style='color:#aaaaaa; font-size:0.68em; margin-top:4px'>{fuente}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.markdown(f"""
                <div class='sector-card'>
                    <div style='font-weight:bold; color:#222222'>{info['nombre']}</div>
                    <div style='color:#888888; font-size:0.72em'>{simbolo}</div>
                    <div style='color:#aaaaaa; margin-top:8px; font-size:0.8em'>Sin datos</div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECCIÓN: MODELO DE PREDICCIÓN
# ─────────────────────────────────────────────

def mostrar_analisis_sector(modelo, scaler, features, lookback, tokenizer, finbert):
    """
    Vista del modelo de predicción — primero la predicción, luego el historial.
    Reorganizada para que el usuario llegue directo a lo que importa.
    """
    st.markdown("## Modelo de predicción")
    st.markdown("""
    <div class='info-box'>
        Selecciona el sector que deseas analizar y pega cualquier texto financiero
        relacionado específicamente con ese sector. El modelo predecirá si subirá
        o bajará mañana. Más abajo encontrarás el historial de precios y la tabla
        de datos históricos.
    </div>
    """, unsafe_allow_html=True)

    # Selector de sector
    opciones    = {f"{v['nombre']} ({k})": k for k, v in SECTORES.items()}
    seleccion   = st.selectbox("Selecciona un sector:", list(opciones.keys()))
    simbolo     = opciones[seleccion]
    info_sector = SECTORES[simbolo]

    st.caption(f"Principales empresas: {info_sector['empresas']}")

    # ── PREDICCIÓN PRIMERO — es lo más importante de esta sección ──
    st.markdown("### Hacer una predicción")

    # Límite aproximado de palabras que procesa FinBERT
    # BERT usa ~1.3 tokens por palabra en inglés financiero
    # 512 tokens / 1.3 ≈ 394 palabras — usamos 380 como margen seguro
    limite_palabras = 380

    st.markdown(f"""
    <div class='info-box'>
        Pega aquí cualquier texto financiero relacionado con el sector de
        <b>{info_sector['nombre']}</b> que seleccionaste — puede ser desde un titular
        corto hasta un artículo completo sobre ese sector específicamente.<br><br>
        El modelo analiza el tono del texto de forma simultánea con el comportamiento
        histórico del precio: mientras FinBERT determina si la noticia es positiva,
        negativa o neutra, el LSTM analiza los últimos 10 días de precio. Ambos análisis
        se combinan para generar la predicción.<br><br>
        Entre más contexto proporciones del sector seleccionado, más información tendrá
        el modelo. El modelo procesa hasta aproximadamente <b>{limite_palabras} palabras</b>.
        Para textos más extensos se recomienda pegar el fragmento más relevante del artículo
        — generalmente el primer o segundo párrafo contiene la información más importante.
    </div>
    """, unsafe_allow_html=True)

    texto_noticia = st.text_area(
        "Texto a analizar:",
        placeholder = (
            f"Pega aquí un titular, párrafo o artículo completo sobre "
            f"{info_sector['nombre']}...\n\n"
            f"Ejemplo: '{info_sector['nombre']} stocks surge as earnings beat expectations "
            f"for the third consecutive quarter, driving investor confidence...'"
        ),
        height = 180,
    )

    # Contador de palabras en tiempo real para que el usuario sepa si excede el límite
    if texto_noticia.strip():
        palabras      = len(texto_noticia.split())
        excede        = palabras > limite_palabras
        color_cnt     = "#c62828" if excede else "#2e7d32"
        aviso         = f"⚠️ Se usarán las primeras {limite_palabras}" if excede else "Dentro del límite"
        st.markdown(
            f"<div style='font-size:0.78em; color:{color_cnt}; text-align:right'>"
            f"{palabras} palabras — {aviso}</div>",
            unsafe_allow_html=True
        )

    if st.button("Generar predicción", use_container_width=True):
        if not texto_noticia.strip():
            st.markdown(
                "<div class='warning-box'>Pega algún texto financiero para continuar.</div>",
                unsafe_allow_html=True
            )
            return

        with st.spinner("Analizando texto y calculando predicción..."):
            clase, confianza, sentimiento, error = predecir(
                simbolo, texto_noticia,
                modelo, tokenizer, finbert,
                scaler, features, lookback
            )

        if clase is None:
            # Explicamos el error con contexto claro para cualquier usuario
            if error == "datos":
                st.markdown("""
                <div class='error-box'>
                    No se pudo obtener suficiente historial de precios para este sector.
                    El modelo necesita al menos 30 días de datos para hacer una predicción.
                    Yahoo Finance no está respondiendo desde el servidor ahora mismo —
                    esto no tiene que ver con el texto que pegaste ni con el modelo.
                    Intenta de nuevo en unos minutos o selecciona otro sector.
                </div>
                """, unsafe_allow_html=True)
            elif error == "features":
                st.markdown("""
                <div class='error-box'>
                    Los datos descargados no fueron suficientes para calcular todas
                    las variables que el modelo necesita. Intenta con otro sector
                    o espera unos minutos.
                </div>
                """, unsafe_allow_html=True)
            return

        # Determinamos nivel de confianza y dirección para elegir recomendación
        nivel     = "alta" if confianza >= 0.60 else ("media" if confianza >= 0.53 else "baja")
        direccion = "neutral" if nivel == "baja" else ("sube" if clase == 1 else "baja")
        rec       = RECOMENDACIONES.get((direccion, nivel), RECOMENDACIONES[("neutral", "baja")])

        # ── RESULTADO VISUAL ──
        col_gif, col_resultado = st.columns([1, 2])

        with col_gif:
            # GIF aleatorio de Giphy según el resultado
            gif_url = obtener_gif_giphy(GIPHY_QUERIES.get((direccion, nivel), "confused stocks"))
            if gif_url:
                st.image(gif_url, use_container_width=True)

        with col_resultado:
            st.markdown(f"""
            <div class='prediction-box' style='border: 2px solid {rec["color"]}'>
                <div style='font-size:0.75em; color:#888888; letter-spacing:2px;
                            text-transform:uppercase; margin-bottom:12px'>
                    Predicción para mañana
                </div>
                <div style='font-size:2.8em; font-weight:800; color:{rec["color"]}'>
                    {"SUBE" if clase == 1 else "BAJA"}
                </div>
                <div style='font-size:0.9em; color:#666666; margin:10px 0'>
                    Confianza del modelo:
                    <b style='color:{rec["color"]}; font-size:1.2em'>{confianza:.1%}</b>
                </div>
                <hr style='border-color:#e0e0e0; margin:14px 0'>
                <div style='font-size:1.05em; color:#333333; margin:12px 0;
                            font-style:italic'>
                    "{rec["frase"]}"
                </div>
                <div style='color:#666666; font-size:0.85em; line-height:1.5'>
                    {rec["accion"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── ANÁLISIS DE SENTIMIENTO DEL TEXTO ──
        st.markdown("#### Qué piensa la IA sobre tu texto")
        col_pos, col_neg, col_neu = st.columns(3)
        with col_pos:
            st.metric("Positivo", f"{sentimiento['positivo']:.1%}")
        with col_neg:
            st.metric("Negativo", f"{sentimiento['negativo']:.1%}")
        with col_neu:
            st.metric("Neutro",   f"{sentimiento['neutro']:.1%}")

        score       = sentimiento["score_neto"]
        color_score = "#2e7d32" if score > 0 else "#c62828"
        st.markdown(f"""
        <div style='margin-top:12px; padding:12px; background:#f8f9fa;
                    border-radius:8px; text-align:center; border:1px solid #e0e0e0'>
            <div style='font-size:0.8em; color:#888888; margin-bottom:6px'>
                Tono general del texto (positivo menos negativo)
            </div>
            <div style='font-size:1.4em; font-weight:700; color:{color_score}'>
                {score:+.3f}
            </div>
            <div style='font-size:0.75em; color:#aaaaaa; margin-top:4px'>
                Escala de -1 (muy negativo) a +1 (muy positivo)
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.caption(
            "Proyecto académico — UANL MCD 2026. No constituye asesoría financiera. "
            "Precisión en datos no vistos: 50%. El mercado siempre tiene la última palabra."
        )

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── GRÁFICA HISTÓRICA — después de la predicción ──
    st.markdown("### Precio histórico")
    st.markdown("""
    <div class='info-box'>
        La gráfica muestra la evolución del precio de este sector durante los últimos 2 años.
        Las marcas amarillas señalan fechas donde el New York Times publicó noticias
        relevantes sobre este sector — se puede observar si los movimientos importantes
        del precio coinciden con días de noticias destacadas.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Descargando historial de precios..."):
        df_hist = obtener_datos_historicos_completos(simbolo)

    if df_hist is not None:
        noticias = obtener_noticias_nyt(QUERIES_NYT.get(simbolo, "stocks"))

        fig = go.Figure()

        # Línea de precio principal con área bajo la curva
        fig.add_trace(go.Scatter(
            x             = df_hist.index,
            y             = df_hist["close"],
            name          = f"{simbolo} precio",
            line          = dict(color="#2196F3", width=2),
            fill          = "tozeroy",
            fillcolor     = "rgba(33, 150, 243, 0.05)",
            hovertemplate = "<b>%{x|%d %b %Y}</b><br>Precio: $%{y:.2f}<extra></extra>"
        ))

        # Anotaciones de noticias usando shapes — más confiable que add_vline en fondos claros
        shapes      = []
        annotations = []
        for noticia in noticias:
            try:
                fecha = pd.to_datetime(noticia["fecha"])
                if df_hist.index.min() <= fecha <= df_hist.index.max():
                    shapes.append(dict(
                        type  = "line",
                        x0    = fecha, x1   = fecha,
                        y0    = 0,     y1   = 1,
                        xref  = "x",   yref = "paper",
                        line  = dict(color="#FFA000", width=2, dash="dot"),
                    ))
                    annotations.append(dict(
                        x         = fecha,
                        y         = 1,
                        xref      = "x",
                        yref      = "paper",
                        text      = "●",
                        showarrow = False,
                        font      = dict(color="#FFA000", size=12),
                        yanchor   = "bottom"
                    ))
            except Exception:
                pass

        fig.update_layout(
            template    = "plotly_white",  # fondo blanco para coincidir con el diseño claro
            height      = 380,
            margin      = dict(l=50, r=20, t=20, b=40),
            xaxis       = dict(title="Fecha"),
            yaxis       = dict(title="Precio (USD)", tickprefix="$"),
            hovermode   = "x unified",
            showlegend  = False,
            shapes      = shapes,
            annotations = annotations,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("● Marcas amarillas = fechas de noticias recientes del NYT sobre este sector")

# ─────────────────────────────────────────────
# SECCIÓN: SOBRE EL PROYECTO
# ─────────────────────────────────────────────

def mostrar_sobre_proyecto():
    """
    Explicación del proyecto en lenguaje accesible para cualquier audiencia.
    Escrita de forma directa y técnica — sin jerga de marketing.
    """
    st.markdown("## Sobre el Proyecto")
    st.markdown("""
    <div class='info-box'>
        Todo lo que necesitas saber sobre cómo funciona esta herramienta,
        qué tan buena es y por qué fue construida así.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Cómo funciona", "Resultados del modelo"])

    with tab1:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("""
            ### Cómo funciona

            Este proyecto combina dos fuentes de información de forma simultánea
            para generar una predicción sobre el comportamiento del mercado al día siguiente.

            **Fuente 1: datos de precio.**
            Se analiza el comportamiento histórico del precio de un sector durante
            los últimos meses. Se calculan indicadores como el rendimiento diario,
            la volatilidad reciente, el volumen de operaciones y el momentum de los
            últimos días — todo a partir de datos reales de Yahoo Finance.

            **Fuente 2: texto financiero.**
            Se toma el texto que proporciona el usuario — un titular o artículo —
            y se analiza con FinBERT, un modelo de procesamiento de lenguaje natural
            entrenado específicamente en textos financieros. Este modelo determina
            si el tono del texto es positivo, negativo o neutro para el mercado.

            **Estas dos fuentes se procesan de forma simultánea y paralela.**
            Los indicadores de precio y los scores de sentimiento del texto se combinan
            en un único vector de 14 variables que entra al modelo LSTM.
            El modelo produce una predicción: ¿subirá o bajará este sector mañana?

            ---

            ### Contexto del proyecto

            Este trabajo fue desarrollado como parte de la Maestría en Ciencia de Datos
            de la UANL, aplicando técnicas de aprendizaje profundo a datos financieros reales.

            El modelo fue entrenado con datos del mercado de **abril 2024 a abril 2026**,
            usando precios de Yahoo Finance y artículos del New York Times.

            Los precios que usa para predecir **sí son en tiempo real** — cada predicción
            descarga los datos más recientes disponibles de Yahoo Finance.

            Sin embargo, **los pesos internos del modelo no se actualizan automáticamente**.
            Para reentrenar el modelo con datos nuevos se requeriría infraestructura de cómputo
            en la nube — por ejemplo un pipeline automatizado en Google Cloud o AWS.
            Para este prototipo académico, el reentrenamiento se realiza de forma manual
            cuando se considera necesario.
            """)

        with col2:
            # Tarjetas técnicas con descripción clara de para qué sirve cada componente
            st.markdown("""
            <div class='tech-card'>
                <div class='tech-card-title' style='color:#2196F3'>LSTM Bidireccional</div>
                <div class='tech-card-label'>Arquitectura del modelo de predicción</div>
                <div class='tech-card-desc'>
                    Red neuronal diseñada para aprender patrones en secuencias de tiempo.
                    Analiza los últimos 10 días de comportamiento del precio antes de generar
                    una predicción. El mecanismo de Attention identifica cuáles días del
                    pasado son más relevantes para la predicción actual.
                    Tiene 148,803 parámetros entrenables.
                </div>
            </div>
            <div class='tech-card'>
                <div class='tech-card-title' style='color:#4CAF50'>FinBERT</div>
                <div class='tech-card-label'>Modelo de análisis de texto financiero</div>
                <div class='tech-card-desc'>
                    Modelo de lenguaje preentrenado en millones de textos financieros.
                    Se encarga de leer el texto que proporciona el usuario y determinar
                    si su tono es positivo, negativo o neutro para el mercado.
                    Su salida — tres probabilidades — forma parte de las 14 variables
                    que recibe el LSTM para hacer la predicción.
                </div>
            </div>
            <div class='tech-card'>
                <div class='tech-card-title' style='color:#f4a261'>4,800+ días de datos</div>
                <div class='tech-card-label'>Base de entrenamiento del modelo</div>
                <div class='tech-card-desc'>
                    El LSTM fue entrenado con datos de 10 sectores del S&P 500
                    durante 2 años — abril 2024 a abril 2026.
                    Cada día de entrenamiento incluye precios de Yahoo Finance
                    y artículos del New York Times procesados con FinBERT.
                    Este es el historial con el que el modelo aprendió a reconocer patrones.
                </div>
            </div>
            <div class='tech-card'>
                <div class='tech-card-title' style='color:#a29bfe'>14 variables por día</div>
                <div class='tech-card-label'>Entradas al modelo en cada predicción</div>
                <div class='tech-card-desc'>
                    Cada predicción combina 14 variables simultáneas: 9 financieras
                    (rendimiento a 1, 5 y 21 días, volatilidad, volumen relativo, rango
                    diario y lags de 3 días) y 5 de sentimiento (score positivo, negativo,
                    neutro, score neto y número de noticias del día).
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Resultados del modelo")
        st.markdown("""
        <div class='info-box'>
            Para entender los resultados no se necesita saber de estadística.
            Se explica con una analogía.
        </div>
        """, unsafe_allow_html=True)

        # Analogía accesible para cualquier tipo de audiencia
        st.markdown("""
        **La analogía del pronosticador del clima**

        Si alguien predice "no lloverá" todos los días, acertará la mayoría de las veces
        en ciudades donde llueve poco — sin saber absolutamente nada del clima real.
        En los mercados pasa algo similar: el mercado sube más días de los que baja,
        por lo que predecir "siempre sube" da un 54% de acierto sin ningún modelo.

        Este modelo va más allá. Aprende a distinguir cuándo es más probable que suba
        y cuándo que baje, basándose en noticias y en el comportamiento reciente del precio.
        Especialmente destaca en detectar bajadas — lo más valioso en finanzas.
        """)

        # Métricas con tarjetas visuales y tooltips explicativos
        c1, c2, c3, c4 = st.columns(4)
        metricas = [
            ("50%", "Precisión general", "#2196F3",
             "De cada 100 predicciones el modelo acierta 50. El punto base sin modelo es 50% "
             "— como lanzar una moneda. Lo que hace valioso a este modelo no es el accuracy "
             "global sino su capacidad diferenciada: detecta bajadas el 69% de las veces, "
             "que es el caso de uso más valioso en finanzas."),
            ("69%", "Detecta bajadas", "#f44336",
             "Cuando el mercado realmente baja, el modelo lo predice correctamente el 69% "
             "de las veces. Este es el número más importante — detectar caídas antes de que "
             "ocurran es lo que más vale en un sistema financiero real."),
            ("35%", "Detecta subidas", "#4CAF50",
             "Cuando el mercado sube, el modelo lo anticipa el 35% de las veces. El modelo "
             "está calibrado para ser conservador con las subidas y más agresivo "
             "detectando bajadas — una estrategia válida en gestión de riesgo."),
            ("10", "Sectores cubiertos", "#f4a261",
             "Los 10 sectores GICS del S&P 500 — el estándar de clasificación de la industria "
             "financiera. Cubren más del 90% del valor del mercado americano."),
        ]
        for col, (val, lbl, color, tooltip) in zip([c1, c2, c3, c4], metricas):
            with col:
                st.markdown(f"""
                <div style='background:#ffffff; border:2px solid #e0e0e0; border-radius:12px;
                            padding:16px; text-align:center; margin-bottom:8px;
                            box-shadow:0 2px 6px rgba(0,0,0,0.06)'>
                    <div style='font-size:2em; font-weight:700; color:{color}'>{val}</div>
                    <div style='font-size:0.72em; color:#888888; margin-top:6px;
                                text-transform:uppercase; letter-spacing:1px'>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(tooltip)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        st.markdown("""
        **Qué significan estas combinaciones en la práctica**

        - **Bajada con alta confianza (>60%):** el caso más valioso. El modelo detecta
          caídas el 69% de las veces — en gestión de riesgo, evitar pérdidas vale más
          que capturar ganancias.

        - **Subida con alta confianza:** señal positiva moderada. El modelo es más
          conservador con las subidas para evitar falsas alarmas.

        - **Confianza baja (~50%):** el modelo no tiene suficiente información para decidir.
          En ese caso la predicción equivale a lanzar una moneda.

        ---

        **Limitación principal**

        Las noticias solo llegan esporádicamente — entre 20 y 100 artículos por sector
        para cubrir dos años de mercado. En producción real, las noticias llegarían
        diariamente desde fuentes como Reuters o Bloomberg, lo que mejoraría
        significativamente la precisión.
        """)

# ─────────────────────────────────────────────
# NAVEGACIÓN PRINCIPAL
# ─────────────────────────────────────────────

def main():
    """
    Función principal — controla el flujo completo de la app.
    Streamlit re-ejecuta main() cada vez que el usuario interactúa.
    session_state persiste valores entre ejecuciones.
    """
    # Inicializamos session_state si es la primera vez que carga la app
    if "bienvenida_vista" not in st.session_state:
        st.session_state["bienvenida_vista"] = False

    # Pantalla de bienvenida — GIF se obtiene aquí, después de que Streamlit
    # inicializó la sesión, para evitar el error "SessionInfo not initialized"
    if not st.session_state["bienvenida_vista"]:
        gif_url = obtener_gif_giphy("stocks wall street trading")
        mostrar_bienvenida(gif_url=gif_url)
        return

    # Carga de modelos — solo ocurre una vez gracias a @st.cache_resource
    with st.spinner("Cargando modelos... (solo la primera vez)"):
        tokenizer, finbert = cargar_finbert()
        modelo, scaler, features, lookback = cargar_modelo_lstm()

    # Barra lateral con identidad del proyecto y botón de regreso
    with st.sidebar:
        st.markdown("## Predictor de Mercados")
        st.markdown("---")
        st.markdown("**Eric Brandon García Luján**")
        st.caption(
            "Herramienta de predicción de movimientos del mercado financiero "
            "usando LSTM, Transformers y análisis de noticias en tiempo real."
        )
        st.markdown("---")
        st.caption("UANL · MCD · 2026")
        st.markdown("---")

        # Botón de regreso a la pantalla inicial
        if st.button("Volver al inicio", use_container_width=True):
            st.session_state["bienvenida_vista"] = False
            st.rerun()

    # Navegación por tabs — más fluida que radio buttons
    tab1, tab2, tab3 = st.tabs([
        "Vista general del mercado",
        "Modelo de predicción",        # renombrado desde "Análisis por Sector"
        "Sobre el Proyecto"
    ])

    with tab1:
        mostrar_dashboard(modelo, scaler, features, lookback, tokenizer, finbert)
    with tab2:
        mostrar_analisis_sector(modelo, scaler, features, lookback, tokenizer, finbert)
    with tab3:
        mostrar_sobre_proyecto()


# Punto de entrada de la app
# Streamlit ejecuta este archivo completo — main() organiza el flujo
if __name__ == "__main__":
    main()
