import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
# ---------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Accidentes de Tránsito - Análisis Estadístico",
    page_icon="🚦",
    layout="wide",
)
 
st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    h1 {color: #ff6b35;}
    h2, h3 {color: #2a9d8f;}
    div[data-testid="stMetric"] {
        background-color: #fff4e6;
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid #ffd8a8;
    }
    div[data-testid="stMetric"] * {
        color: #1a1a1a !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #7a4a1f !important;
    }
    div[data-testid="stMetricValue"] * {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #d0ebff;
    }
    section[data-testid="stSidebar"] {
        background-color: #8a8580;
    }
</style>
""", unsafe_allow_html=True)
 
# ---------------------------------------------------------------
# Carga de datos (igual que en el notebook)
# ---------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Accidentes de Transito - Dataset.csv", encoding="utf-8-sig")
    return df
 
df = cargar_datos()
 
# ---------------------------------------------------------------
# Navegación lateral (sigue el orden del notebook resumen)
# ---------------------------------------------------------------
st.sidebar.title("🚦 Navegación")
seccion = st.sidebar.radio(
    "Ir a la sección:",
    [
        "Información del dataset",
        "Variable objetivo: Accidentes",
        "Variables independientes",
        "Modelo de regresión lineal múltiple",
        "Predicción interactiva",
    ],
)
 
# =================================================================
# SECCIÓN 1 — INFORMACIÓN DEL DATASET (celdas 2-8 del notebook)
# =================================================================
if seccion == "Información del dataset":
    st.title("🚦 Accidentes de Tránsito — Información del conjunto de datos")
 
    st.markdown(
        """
        En este proyecto se excluyó el periodo comprendido entre marzo de 2020 y agosto de 2021,
        correspondiente a la pandemia de COVID-19. Se consideró que este intervalo no representa
        condiciones normales de tráfico, ya que las restricciones de movilidad y demás medidas
        sanitarias generan un comportamiento atípico en la accidentalidad. Por esta razón, el análisis
        se centra únicamente en periodos que reflejan condiciones normales de circulación vehicular.
 
        Las variables Días Hábiles del mes e Inicio de Clases fueron calculadas manualmente con base
        en el calendario oficial de cada año.
        """
    )
 
    st.subheader("Vista general del dataset")
    st.dataframe(df.head(), use_container_width=True)
 
    st.subheader("Dimensiones, tipos de dato y nulos")
    info_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo (dtype)": df.dtypes.astype(str).values,
        "Nulos": df.isnull().sum().values,
    })
    st.table(info_df)
 
    st.subheader("Resumen estadístico (solo variables numéricas)")
    st.dataframe(df.describe(), use_container_width=True)
 
    st.subheader("Tipos de variables")
    tipos_variables = {
        "Mes": "Cualitativa ordinal",
        "Año": "Cuantitativa discreta",
        "Accidentes": "Cuantitativa discreta",
        "Temperatura Prom. del Mes (°C)": "Cuantitativa continua",
        "Acum. Lluvia Diaria": "Cuantitativa continua",
        "Días Hábiles del mes": "Cuantitativa discreta",
        "Inicio de Clases": "Cualitativa dicotómica",
        "Precio Gasolina (95 Octanos)": "Cuantitativa continua",
        "Accidentes del mes anterior": "Cuantitativa discreta",
        "IMAE": "Cuantitativa continua",
    }
    st.table(pd.DataFrame(tipos_variables.items(), columns=["Variable", "Tipo"]))
 
    st.markdown(
        """
        Con los tipos de variables ya identificados, se puede determinar dónde tiene sentido aplicar
        los métodos estadísticos correspondientes en este proyecto.
 
        Para Accidentes de Tránsito se realizará un análisis más profundo, dado que es la variable
        objetivo (target) del proyecto.
 
        Para las demás variables:
 
        **Cuantitativas discretas y continuas:**
        Temperatura Prom. del Mes (°C), Acum. Lluvia Diaria, Precio Gasolina (95 Octanos), IMAE.
        Se les aplicarán medidas de tendencia central (media, mediana, moda), medidas de dispersión
        (rango, varianza, desviación estándar, coeficiente de variación), coeficiente de asimetría
        de Pearson, curtosis, e histograma con su respectivo polígono de frecuencia.
        """
    )
 
    st.info(
        """
        **Nota metodológica:**
        Las variables Mes, Inicio de Clases y Accidentes del mes anterior fueron excluidas del análisis
        exploratorio detallado. Mes es una variable cíclica que se repite cada 12 observaciones a lo largo
        de todo el dataset (aproximadamente 7 a 9 veces cada categoría), por lo que su distribución de
        frecuencias es prácticamente uniforme y no aporta información relevante por sí sola; su efecto
        real sobre la accidentalidad se captura mejor de forma indirecta a través de variables que varían
        con el mes, como Temperatura, Lluvia y Días Hábiles. De manera similar, Inicio de Clases es una
        variable dicotómica fuertemente desbalanceada (84% en la categoría 0 y 16% en la categoría 1), lo
        que limita su capacidad explicativa como variable independiente en un análisis estadístico
        descriptivo; su relación con Accidentes se evaluará más adelante de forma puntual, comparando el
        promedio de accidentes entre ambas.
 
        Por último, Accidentes del mes anterior tampoco se analiza en profundidad, ya que es una variable
        rezagada (lag) de la propia variable objetivo: sus valores son simplemente los Accidentes de la
        observación anterior desplazados un mes, por lo que su distribución es prácticamente idéntica a la
        de Accidentes y su análisis estadístico individual sería redundante. Su relevancia real no está en
        su distribución, sino en su relación temporal con el target (autocorrelación), la cual se abordará
        más adelante como parte del análisis de relación entre variables.
        """
    )
 
# =================================================================
# SECCIÓN 2 — VARIABLE OBJETIVO: ACCIDENTES (celdas 9-17)
# =================================================================
elif seccion == "Variable objetivo: Accidentes":
    st.title("🎯 Accidentes de Tránsito (variable objetivo)")
 
    # ---- Histograma y polígono de frecuencia (idéntico al notebook) ----
    st.subheader("Histograma y polígono de frecuencia")
 
    frecuencias, intervalos = np.histogram(df["Accidentes"], bins=8)
 
    bins = []
    for index, value in enumerate(intervalos):
        if index <= 7:
            bins.append((int(value), int(intervalos[index + 1])))
 
    Marcas_de_clase = [(a + b) / 2 for a, b in bins]
    labels = [f"({a}-{b})" for a, b in bins]
    Anchura = (bins[0][1] - bins[0][0])
 
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(Marcas_de_clase, frecuencias, width=Anchura, align="center",
           edgecolor="black", color="skyblue", alpha=0.7, label="Frecuencia")
    ax.plot(Marcas_de_clase, frecuencias, marker=".", markersize=10,
            markerfacecolor="#a32a51", markeredgecolor="#a32a51", linestyle="solid",
            linewidth=2, color="orange", label="Polígono de frecuencias")
    ax.set_xticks(Marcas_de_clase)
    ax.set_xticklabels(labels)
    ax.set_title("Histograma y Polígono de frecuencia")
    ax.set_xlabel("Número de Accidentes")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
 
    st.markdown(
        """
        El histograma muestra que la mayoría de los meses presentan entre aproximadamente 2251 y 2643
        accidentes, concentración que coincide con el intervalo de mayor frecuencia (18 meses en cada uno
        de los dos intervalos centrales). La distribución cae de forma bastante simétrica hacia ambos lados
        de ese centro. Los valores extremos son poco frecuentes: 6 meses registran entre 1663 y 1859
        accidentes, y apenas 4 meses superan los 3035. Esto sugiere una distribución simétrica.
        """
    )
 
    # ---- Tendencia central ----
    st.subheader("Medidas de tendencia central")
    media = df["Accidentes"].mean()
    mediana = df["Accidentes"].median()
    moda = df["Accidentes"].mode()
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Media", f"{media:.2f}")
    c2.metric("Mediana", f"{mediana}")
    c3.metric("Moda(s)", ", ".join(str(m) for m in moda))
 
    st.markdown(
        """
        La media de 2440.47 y la mediana de 2435.0 son muy cercanas entre sí, lo que confirma la simetría
        observada en el histograma: no hay indicios de sesgo marcado ni de valores extremos que distorsionen
        el promedio. También se puede observar que la variable accidentes es multimodal (1967, 2269 y 2457),
        con cada valor repitiéndose solo 2 veces, esto significa que pierde utilidad como medida de tendencia
        central, ya que no logra capturar un "valor típico".
        """
    )
 
    # ---- Dispersión ----
    st.subheader("Medidas de dispersión")
    rango = df["Accidentes"].max() - df["Accidentes"].min()
    varianza = df["Accidentes"].var(ddof=0)
    std0 = df["Accidentes"].std(ddof=0)
    cv = (df["Accidentes"].std() / df["Accidentes"].mean()) * 100
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rango", f"{rango:.2f}")
    c2.metric("Varianza", f"{varianza:.2f}")
    c3.metric("Desviación estándar", f"{std0:.2f}")
    c4.metric("Coef. de variación", f"{cv:.2f}%")
 
    st.markdown(
        """
        La variable Accidentes presenta una desviación estándar de 362.80, lo que indica que los valores
        mensuales se alejan en promedio ±363 accidentes respecto a la media de 2440.47. El coeficiente de
        variación (14.95%) confirma que esta dispersión es baja-moderada en términos relativos, sugiriendo
        que, si bien existen fluctuaciones mes a mes, el comportamiento de la accidentalidad es relativamente
        estable a lo largo del periodo analizado, sin cambios extremos o erráticos.
        """
    )
 
    # ---- Asimetría y curtosis ----
    st.subheader("Asimetría de Pearson y curtosis")
    asimetria = (3 * (media - mediana)) / df["Accidentes"].std(ddof=0)
    curt = kurtosis(df["Accidentes"])
 
    c1, c2 = st.columns(2)
    c1.metric("Asimetría de Pearson", f"{asimetria:.3f}")
    c2.metric("Curtosis", f"{curt:.3f}")
 
    st.markdown(
        """
        El coeficiente de asimetría de Pearson de 0.045 confirma que la distribución de Accidentes es
        prácticamente simétrica, sin sesgo relevante hacia ningún lado, resultado consistente con la cercanía
        observada entre la media y la mediana. Por su parte, la curtosis de -0.684 indica una distribución
        platicúrtica: más plana que una distribución normal, con los datos repartidos de forma más uniforme a
        lo largo del rango y sin una concentración marcada alrededor del valor central. En conjunto, estos
        resultados sugieren que la accidentalidad mensual se comporta de forma bastante estable y predecible,
        sin sesgos ni comportamientos extremos que distorsionen la distribución.
        """
    )
 
# =================================================================
# SECCIÓN 3 — VARIABLES INDEPENDIENTES (celdas 18-30)
# =================================================================
elif seccion == "Variables independientes":
    st.title("📊 Análisis descriptivo de las variables independientes")
 
    st.markdown(
        """
        Además de la variable objetivo (Accidentes), el proyecto considera un conjunto de variables
        independientes que podrían influir en la frecuencia de accidentes de tránsito en la provincia de
        Panamá.
 
        En esta sección se presenta un análisis descriptivo de las principales variables cuantitativas
        utilizadas en el modelo de regresión lineal múltiple: temperatura promedio del mes, acumulación de
        lluvia diaria, precio de la gasolina de 95 octanos e Índice Mensual de Actividad Económica (IMAE).
 
        Para cada variable se calcularon medidas de dispersión y cuartiles con el objetivo de comprender su
        comportamiento, evaluar su variabilidad y conocer cómo se distribuyen los datos antes de desarrollar
        el modelo predictivo.
        """
    )
 
    variables_info = {
        "Temperatura Prom. del Mes (°C)": {
            "col": "Temperatura Prom. del Mes (°C)",
            "intro": (
                "La temperatura promedio mensual es una variable climática que representa las condiciones "
                "ambientales registradas durante cada mes del periodo analizado. Su estudio permite determinar "
                "el grado de variabilidad de la temperatura y posteriormente evaluar si existe alguna relación "
                "con la frecuencia de accidentes de tránsito."
            ),
            "interpretacion": (
                "Los resultados muestran que la temperatura promedio mensual presenta una variabilidad "
                "relativamente baja durante el período de estudio. Este comportamiento es consistente con las "
                "características climáticas de Panamá, donde las temperaturas suelen mantenerse estables a lo "
                "largo del año.\n\nLos cuartiles permiten observar cómo se distribuyen los valores de temperatura "
                "dentro del conjunto de datos y sirven como referencia para comparar los distintos períodos "
                "analizados. Aunque la temperatura no presenta cambios tan pronunciados como otras variables, se "
                "incluye dentro del modelo para evaluar su posible influencia sobre la cantidad de accidentes de "
                "tránsito."
            ),
        },
        "Acum. Lluvia Diaria": {
            "col": "Acum. Lluvia Diaria",
            "intro": (
                "La acumulación de lluvia diaria representa el nivel de precipitación registrado durante el "
                "período analizado. Debido a que las condiciones meteorológicas pueden afectar la visibilidad y "
                "el estado de las vías, esta variable constituye uno de los factores de mayor interés dentro del "
                "estudio."
            ),
            "interpretacion": (
                "La acumulación de lluvia presenta una variabilidad mayor que la observada en la temperatura "
                "promedio, lo cual refleja las diferencias entre la estación seca y la estación lluviosa en "
                "Panamá.\n\nEl análisis de los cuartiles permite identificar la distribución de los niveles de "
                "precipitación durante el período estudiado. Debido a la influencia que la lluvia puede tener "
                "sobre las condiciones de conducción, esta variable resulta especialmente importante para el "
                "desarrollo del modelo de regresión lineal múltiple."
            ),
        },
        "Precio Gasolina (95 Octanos)": {
            "col": "Precio Gasolina (95 Octanos)",
            "intro": (
                "El precio de la gasolina representa un indicador económico que puede influir indirectamente "
                "sobre la movilidad vehicular. Su inclusión dentro del análisis permite evaluar si las "
                "variaciones en el costo del combustible presentan alguna relación con la frecuencia de "
                "accidentes registrados."
            ),
            "interpretacion": (
                "El precio de la gasolina presenta variaciones moderadas durante el período analizado. Aunque "
                "esta variable no describe directamente el comportamiento de los accidentes, puede estar "
                "relacionada con la cantidad de desplazamientos realizados por la población y, por consiguiente, "
                "con la exposición al riesgo de accidentes.\n\nSu incorporación al modelo permitirá evaluar "
                "estadísticamente si existe alguna relación significativa entre ambas variables."
            ),
        },
        "IMAE": {
            "col": "IMAE",
            "intro": (
                "El Índice Mensual de Actividad Económica (IMAE) constituye un indicador del comportamiento de "
                "la economía nacional. En este proyecto se incorpora como una variable independiente con el "
                "propósito de analizar si el nivel de actividad económica puede influir sobre la frecuencia de "
                "accidentes de tránsito."
            ),
            "interpretacion": (
                "El IMAE presenta variaciones graduales durante el período estudiado, reflejando cambios en la "
                "actividad económica del país. La información obtenida mediante las medidas de dispersión y los "
                "cuartiles permite comprender el comportamiento de esta variable antes de incorporarla al modelo "
                "de regresión.\n\nSu análisis resulta relevante debido a que un incremento en la actividad "
                "económica podría asociarse con una mayor movilidad de personas y vehículos, lo que eventualmente "
                "podría influir en la cantidad de accidentes registrados."
            ),
        },
    }
 
    tabs = st.tabs(list(variables_info.keys()))
    for tab, (nombre, info) in zip(tabs, variables_info.items()):
        with tab:
            st.markdown(info["intro"])
            serie = df[info["col"]]
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Rango", f"{serie.max() - serie.min():.4f}")
            c2.metric("Varianza", f"{serie.var():.4f}")
            c3.metric("Desviación estándar", f"{serie.std():.4f}")
 
            q = serie.quantile([0.25, 0.50, 0.75])
            st.markdown("**Cuartiles**")
            st.table(q.rename("Valor"))
 
            st.markdown("### Interpretación")
            st.markdown(info["interpretacion"])
 
# =================================================================
# SECCIÓN 4 — MODELO DE REGRESIÓN LINEAL MÚLTIPLE (celdas 31-38)
# =================================================================
elif seccion == "Modelo de regresión lineal múltiple":
    st.title("📈 Modelo de regresión lineal múltiple")
 
    meses_num = {'Enero':1,'Febrero':2,'Marzo':3,'Abril':4,'Mayo':5,'Junio':6,
                 'Julio':7,'Agosto':8,'Septiembre':9,'Octubre':10,'Noviembre':11,'Diciembre':12}
    df_corr = df.copy()
    df_corr['Mes'] = df_corr['Mes'].map(meses_num)
 
    st.subheader("Mapa de calor de correlaciones")
    corr = df_corr.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
 
    st.markdown(
        """
        La accidentalidad se explica principalmente por su propia inercia temporal (Accidentes del mes
        anterior, r=0.70) y por factores económicos: Precio Gasolina (r=-0.65) e IMAE (r=0.55), mientras que
        las variables climáticas (Temperatura y Lluvia) no muestran relación relevante con el target.
 
        Un hallazgo adicional relevante es que estas mismas variables económicas están fuertemente
        correlacionadas entre sí (Precio Gasolina vs. IMAE, r=-0.77; Año vs. IMAE, r=-0.75), lo que sugiere
        que no aportan información completamente independiente y podrían generar multicolinealidad si se usan
        juntas en un modelo predictivo.
        """
    )
 
    st.subheader("Accidentes de Tránsito vs. cada variable")
    features_scatter = ['Mes', 'Año', 'Temperatura Prom. del Mes (°C)', 'Acum. Lluvia Diaria',
                         'Días Hábiles del mes', 'Inicio de Clases', 'Precio Gasolina (95 Octanos)',
                         'Accidentes del mes anterior', 'IMAE']
    fig2, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, feature in enumerate(features_scatter):
        axes[i].scatter(df_corr[feature], df_corr['Accidentes'], alpha=0.6, color='steelblue', edgecolor='black')
        axes[i].set_xlabel(feature, fontsize=9)
        axes[i].set_ylabel('Accidentes', fontsize=9)
        axes[i].grid(alpha=0.3)
    plt.suptitle('Accidentes de Tránsito vs. cada variable', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2)
 
    st.markdown(
        """
        Los diagramas de dispersión confirman visualmente los hallazgos de la matriz de correlación:
        Accidentes del mes anterior, Precio Gasolina e IMAE muestran relaciones claras con la variable
        objetivo, mientras que Temperatura y Lluvia no presentan ningún patrón.
        """
    )
 
    st.subheader("Resultados del modelo")
    features = ['Accidentes del mes anterior', 'Precio Gasolina (95 Octanos)', 'Año']
    X = df[features]
    y = df['Accidentes']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
 
    st.code(
        f"MAE:  {mae:.0f} accidentes\n"
        f"RMSE: {rmse:.0f} accidentes\n"
        f"R² por fold: {scores}\n"
        f"R² promedio (CV): {scores.mean():.3f} ± {scores.std():.3f}"
    )
 
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.0f} accidentes")
    c2.metric("RMSE", f"{rmse:.0f} accidentes")
    c3.metric("R² promedio (CV)", f"{scores.mean():.3f} ± {scores.std():.3f}")
 
    st.markdown(
        """
        El modelo de regresión lineal, evaluado mediante validación cruzada (5 folds), explica en promedio
        un 48% de la variabilidad de Accidentes (R² = 0.480 ± 0.103), con un error absoluto promedio de 171
        accidentes y un error cuadrático medio de 212 accidentes. Esto indica un desempeño moderado: el
        modelo captura una parte importante de la relación entre las variables predictoras y la
        accidentalidad, pero aún queda una porción considerable de la variabilidad sin explicar, probablemente
        asociada a factores no incluidos en el modelo o a la naturaleza no completamente lineal de algunas
        relaciones.
        """
    )
 
# =================================================================
# SECCIÓN 5 — PREDICCIÓN INTERACTIVA
# =================================================================
elif seccion == "Predicción interactiva":
    st.title("🔮 Predicción interactiva")
    st.markdown(
        "Ingresa valores para las tres variables predictoras del modelo (las mismas usadas en el notebook) "
        "y obtén una predicción del número de Accidentes de Tránsito."
    )
 
    features = ['Accidentes del mes anterior', 'Precio Gasolina (95 Octanos)', 'Año']
    X = df[features]
    y = df['Accidentes']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        accidentes_mes_anterior = st.number_input(
            "Accidentes del mes anterior",
            min_value=int(df["Accidentes del mes anterior"].min()),
            max_value=int(df["Accidentes del mes anterior"].max()) + 500,
            value=int(df["Accidentes del mes anterior"].median()),
            step=10,
        )
    with col2:
        precio_gasolina = st.number_input(
            "Precio Gasolina (95 Octanos)",
            min_value=0.0,
            max_value=3.0,
            value=float(df["Precio Gasolina (95 Octanos)"].median()),
            step=0.01,
            format="%.2f",
        )
    with col3:
        anio = st.number_input(
            "Año",
            min_value=int(df["Año"].min()),
            max_value=int(df["Año"].max()) + 5,
            value=int(df["Año"].max()),
            step=1,
        )
 
    if st.button("Calcular predicción", type="primary"):
        entrada = pd.DataFrame([[accidentes_mes_anterior, precio_gasolina, anio]], columns=features)
        prediccion = model.predict(entrada)[0]
 
        st.metric("Accidentes de Tránsito estimados", f"{prediccion:.0f}")
 
    st.caption(
        "El modelo utilizado aquí es el mismo entrenado en la sección "
        "'Modelo de regresión lineal múltiple' (train/test split 80/20, random_state=42)."
    )
 
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Basado en el notebook resumen del proyecto de Accidentes de Tránsito.")