# Predicción de Accidentes de Tránsito en la Provincia de Panamá

**Universidad Tecnológica de Panamá**  
Facultad de Ingeniería de Sistemas Computacionales  
Departamento de Programación de Computadoras  
Grupo: ISF121

**Integrantes:**
- Diosa Moreno
- Evan Pardo
- Katleen Rujano

---

## Descripción

Este proyecto desarrolla un dashboard de análisis estadístico y un modelo de regresión lineal múltiple para identificar patrones y predecir la frecuencia de accidentes de tránsito en la provincia de Panamá durante el período 2017–2026.

Los datos provienen del Instituto Nacional de Estadística y Censo (INEC) y del Instituto de Meteorología e Hidrología (IMH), integrando variables históricas de accidentalidad con factores meteorológicos, temporales y geográficos.

---

## Metodología

1. **Estadística descriptiva** — distribuciones de frecuencia, medidas de tendencia central y dispersión
2. **Análisis de correlación** — correlación de Pearson entre variables
3. **Regresión lineal múltiple** — modelo predictivo de frecuencia de accidentes
4. **Evaluación del modelo** — R², RMSE y MAE
5. **Dashboard interactivo** — visualización de resultados y hallazgos

---

## Variables del Modelo

| Variable | Fuente |
|---|---|
| Accidentes mensuales | INEC |
| Temperatura promedio mensual (°C) | IMH |
| Acumulado de lluvia diaria (mm) | IMH |
| Días hábiles del mes | Calculado |
| Inicio de clases | Calendario escolar |
| Precio de gasolina (95 octanos) | Registros históricos ACODECO |
| Uso de transporte público (MiBus, Metro) | INEC |

> **Nota:** El período COVID-19 (2020–2021) fue excluido del análisis por representar condiciones atípicas que distorsionan las correlaciones normales.

---

## Cómo Empezar

### 1. Clonar el repositorio
```bash
git clone https://github.com/Mikael-NoFeik/accidentes-transito-panama.git
cd accidentes-transito-panama
```

### 2. Crear y activar el entorno virtual
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Tecnologías Utilizadas

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-Learn
- Streamlit
- Jupyter Notebook
- Visual Studio Code
- Microsoft Excel

---

## Fuentes de Datos

- [Instituto Nacional de Estadística y Censo (INEC)](https://www.inec.gob.pa)
- [Instituto de Meteorología e Hidrología (IMH)](https://www.imhpa.gob.pa)