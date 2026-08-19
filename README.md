# Predictor de Mercados 

Aplicación de deep learning multimodal que estima la dirección del día siguiente (sube/baja) de los diez ETFs sectoriales del S&P 500, combinando datos de precio en tiempo real con el sentimiento de noticias financieras. Construida con una red LSTM bidireccional propia (PyTorch) y FinBERT, desplegada en Hugging Face Spaces.

** Demo:** [Hugging Face Space](https://huggingface.co/spaces/brandonglujan/MCD_BigData_2026)


## Qué hace

El usuario elige uno de los diez sectores del S&P 500 y pega cualquier texto financiero sobre él (un titular, un párrafo, un artículo completo). La app entonces:

1. Descarga el historial de precios reciente del sector desde Yahoo Finance (`yfinance`).
2. Evalúa el tono del texto con **FinBERT** (positivo / negativo / neutro).
3. Combina ambas señales en un solo vector de variables y lo alimenta a una **LSTM bidireccional con mecanismo de atención** para predecir si el sector subirá o bajará al día siguiente.

También muestra una gráfica de precios de dos años anotada con las fechas en que el New York Times publicó cobertura relevante, para inspeccionar visualmente si los movimientos importantes del precio coinciden con días de noticias.

## Cómo funciona el modelo

La idea central es la **fusión multimodal**: el comportamiento del precio y el sentimiento de las noticias se procesan juntos, no por separado.

- **Variables de precio (9):** rendimientos a 1, 5 y 21 días, volatilidad a 10 días, volumen relativo, rango diario y tres rendimientos rezagados (lags).
- **Variables de sentimiento (5):** probabilidades de FinBERT (positivo / negativo / neutro), un score de sentimiento neto y un indicador de número de noticias.
- Estas 14 variables por día alimentan una LSTM bidireccional (~149K parámetros entrenables). Una capa de atención pondera cuáles de los últimos 10 días importan más para la predicción.

La arquitectura y el entrenamiento están en PyTorch; FinBERT (`ProsusAI/finbert`) se carga desde el Hugging Face Hub en tiempo de ejecución.

## Resultados 

El modelo se entrenó con ~2 años de datos (abril 2024 – abril 2026) sobre los 10 sectores.

| Métrica | Valor |
|---|---|
| Precisión general (accuracy) | ~50% |
| Baseline naïve ("siempre sube") | ~54% |
| Recall en días a la baja | ~69% |
| Recall en días al alza | ~35% |

Predecir la *dirección* diaria del mercado es notoriamente casi aleatorio, y estos números lo reflejan con honestidad: el modelo **NO** le gana al baseline naïve en accuracy bruto. Su valor está en otro lado, está deliberadamente calibrado para detectar **caídas** (69% de recall en días a la baja), que es el caso más útil en gestión de riesgo, y el proyecto en conjunto demuestra un pipeline multimodal completo, reproducible y desplegado, evaluado contra baselines adecuados en lugar de contra una métrica escogida a conveniencia.

## Stack tecnológico

`Python` · `PyTorch` · `Transformers / FinBERT` · `scikit-learn` · `Streamlit` · `Plotly` · `yfinance` · `Hugging Face Spaces`

## Estructura del repositorio

```
├── src/
│   ├── streamlit_app.py      # app de Streamlit + definición del modelo e inferencia
│   ├── modelo_lstm.pt        # pesos entrenados de la LSTM (PyTorch)
│   ├── scaler.pkl            # scaler de variables ajustado
│   ├── hiperparametros.json  # hiperparámetros de la arquitectura
│   └── features.json         # lista ordenada de variables
├── requirements.txt
└── README.md
```

## Cómo correrlo localmente

```bash
# 1. Clonar
git clone https://github.com/brandonglujan/Predictor-de-mercados.git
cd Predictor-de-mercados

# 2. Entorno
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Secrets (usa tus propias API keys gratuitas)
export NYT_API_KEY="tu_key_del_nyt"
export GIPHY_API_KEY="tu_key_de_giphy"

# 4. Ejecutar
streamlit run src/streamlit_app.py
```

Las keys de NYT y Giphy son opcionales para la predicción principal — solo alimentan las anotaciones de noticias y los GIFs del resultado. FinBERT y los datos de precio funcionan sin ellas.

## Fuentes de datos

- **Precios:** Yahoo Finance vía `yfinance` (los 10 ETFs sectoriales GICS del S&P 500).
- **Noticias:** API de búsqueda de artículos del New York Times (solo para las anotaciones de la gráfica).
- **Modelo de sentimiento:** `ProsusAI/finbert`, preentrenado en texto financiero.

## Limitaciones

La cobertura de noticias es escasa (entre 20 y 100 artículos por sector en dos años), lo que limita cuánto puede aportar la señal de sentimiento. En un entorno de producción, un feed diario desde una fuente como Reuters o Bloomberg fortalecería el modelo de forma notable. Los pesos tampoco se reentrenan automáticamente; actualizarlos requiere una corrida de entrenamiento manual (o programada en la nube).

## Autor

**Eric Brandon García Luján**
Maestría en Ciencia de Datos — Universidad Autónoma de Nuevo León (UANL), 2026
[LinkedIn](https://www.linkedin.com/in/brandongarcialujan) · [GitHub](https://github.com/brandonglujan)
