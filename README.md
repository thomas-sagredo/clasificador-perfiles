# 🧪 Clasificador de Perfiles de Estudio

Mini-encuesta interactiva que usa **Regresión Logística** para clasificar perfiles de estudio en dos categorías:
- 📚 **Estratégicos**: Planifican, organizan y usan estrategias estructuradas
- 💡 **Intuitivos**: Aprenden por práctica y respuestas rápidas

---

## 🚀 Instalación Rápida

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

O instalar manualmente:
```bash
pip install streamlit scikit-learn pandas numpy joblib plotly
```

### 2. Ejecutar la aplicación
```bash
streamlit run streamlit_logistic_study_profile.py
```

La app se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📋 Características

### ✨ Funcionalidades principales

1. **📊 Dataset & Modelo**
   - Vista previa del dataset
   - Métricas de accuracy
   - Matriz de confusión interactiva
   - Reporte de clasificación completo

2. **🎯 Encuesta Interactiva**
   - 7 preguntas sobre hábitos de estudio
   - Predicción en tiempo real
   - Gauge visual de probabilidad
   - Análisis de contribución de cada respuesta

3. **📈 Análisis Avanzado**
   - Importancia de variables (coeficientes)
   - Distribución de perfiles
   - Matriz de correlación
   - Estadísticas por perfil

4. **💾 Exportar**
   - Descargar modelo entrenado (.pkl)
   - Descargar dataset simulado (CSV)
   - Plantilla de encuesta para recolectar datos

---

## 🎯 Preguntas de la Encuesta

Las 7 preguntas que clasifican tu perfil:

| Variable | Pregunta | Impacto |
|----------|----------|---------|
| 📅 `planifica` | ¿Planificás tu semana de estudio con anticipación? | ⬆️ Estratégico |
| 📱 `usa_apps` | ¿Usás herramientas digitales para organizarte? | ⬆️ Estratégico |
| 🧑 `estudia_solo` | ¿Preferís estudiar solo/a antes que en grupo? | ➡️ Neutral |
| 🌐 `consulta_fuentes` | ¿Consultás fuentes externas (videos, IA, foros)? | ⬆️ Estratégico |
| ✍️ `prefiere_practica` | ¿Aprendés más resolviendo ejercicios que leyendo? | ⬆️ Estratégico |
| ⏰ `procrastina` | ¿Sos de dejar todo para último momento? | ⬇️ Intuitivo |
| 📝 `usa_resumenes` | ¿Te resulta útil repasar con resúmenes o mapas? | ⬆️ Estratégico |

---

## 📁 Estructura del Proyecto

```
tpregresion/
│
├── streamlit_logistic_study_profile.py  # 🎯 App principal (Streamlit)
├── clasificador_cli.py                   # 🖥️ Versión línea de comandos
├── analisis_perfiles_estudio.ipynb      # 📓 Notebook Jupyter completo
├── generar_graficos_presentacion.py     # 📊 Generador de gráficos
│
├── requirements.txt                      # 📦 Dependencias
├── README.md                             # 📖 Este archivo
├── GUIA_PRESENTACION.md                 # 🎤 Guía para presentar en clase
│
└── (generados al ejecutar)
    ├── modelo_perfiles_estudio.pkl      # 🤖 Modelo entrenado
    ├── dataset_perfiles_estudio.csv     # 📊 Dataset de ejemplo
    ├── plantilla_encuesta.csv           # 📋 Plantilla para recolectar datos
    └── graficos_presentacion/           # 🎨 Gráficos para PowerPoint
        ├── 1_distribucion_perfiles.png
        ├── 2_importancia_variables.png
        ├── 3_matriz_confusion.png
        ├── 4_curva_roc.png
        ├── 5_funcion_sigmoide.png
        ├── 6_matriz_correlacion.png
        ├── 7_distribucion_variables.png
        ├── 8_metricas_modelo.png
        └── 9_pipeline.png
```

---

## 🎓 Uso en Clase

### Opción 1: Demo con dataset simulado
1. Ejecutar la app sin cargar CSV
2. Usar el dataset simulado (300 registros)
3. Probar la encuesta interactiva
4. Mostrar análisis y visualizaciones

### Opción 2: Recolectar datos reales
1. Descargar la plantilla CSV desde la app
2. Distribuir a los compañeros para completar
3. Recolectar respuestas (formato: 0=No, 1=Sí)
4. Cargar el CSV en la barra lateral
5. Entrenar el modelo con datos reales

### Opción 3: Usar modelo pre-entrenado
1. Descargar el modelo (.pkl) desde la app
2. Usar en otros scripts o notebooks:

```python
import joblib
import numpy as np

# Cargar modelo
modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
model = modelo_data['model']
scaler = modelo_data['scaler']

# Hacer predicción
# Formato: [planifica, usa_apps, estudia_solo, consulta_fuentes, 
#           prefiere_practica, procrastina, usa_resumenes]
respuestas = np.array([[1, 1, 0, 1, 1, 0, 1]])
respuestas_scaled = scaler.transform(respuestas)
prediccion = model.predict(respuestas_scaled)[0]
probabilidad = model.predict_proba(respuestas_scaled)[0][1]

print(f"Perfil: {'Estratégico' if prediccion == 1 else 'Intuitivo'}")
print(f"Probabilidad: {probabilidad:.2%}")
```

---

## 🎨 Personalización

### Configuración en la barra lateral:
- ✅ Cargar CSV propio
- ✅ Mostrar estadísticas detalladas
- ✅ Mostrar importancia de variables

### Modificar preguntas:
Editar el diccionario `preguntas` en el código (línea ~192):
```python
preguntas = {
    'tu_variable': ('🎯', '¿Tu pregunta personalizada?'),
    # ...
}
```

---

## 📊 Interpretación de Resultados

### Probabilidad de ser Estratégico:
- **> 80%**: Muy probable estratégico
- **60-80%**: Probablemente estratégico
- **40-60%**: Perfil mixto
- **< 40%**: Más probable intuitivo

### Coeficientes del modelo:
- **Positivos**: Aumentan probabilidad de ser estratégico
- **Negativos**: Disminuyen probabilidad (más intuitivo)
- **Magnitud**: Indica importancia de la variable

---

## 🖥️ Versión CLI (Línea de Comandos)

Si preferís una versión sin interfaz gráfica:

```bash
# Modo interactivo (menú)
python clasificador_cli.py

# Entrenar modelo directamente
python clasificador_cli.py --entrenar

# Predecir con modelo existente
python clasificador_cli.py --predecir

# Entrenar con CSV propio
python clasificador_cli.py --csv mis_datos.csv
```

### Características de la versión CLI:
- ✅ Sin dependencias de Streamlit
- ✅ Menú interactivo
- ✅ Entrenamiento y predicción
- ✅ Ideal para servidores o scripts automatizados

---

## 📓 Notebook Jupyter

Para análisis paso a paso con explicaciones didácticas:

```bash
jupyter notebook analisis_perfiles_estudio.ipynb
```

### Contenido del notebook:
1. Importación de librerías
2. Generación y exploración de datos
3. Análisis exploratorio (EDA)
4. Preprocesamiento
5. Entrenamiento del modelo
6. Evaluación con múltiples métricas
7. Predicción con ejemplos
8. Guardado del modelo
9. Conclusiones

---

## 🎤 Presentación en Clase

### Generar gráficos para PowerPoint:

```bash
python generar_graficos_presentacion.py
```



Esto creará 9 gráficos profesionales en la carpeta `graficos_presentacion/`:
- Distribución de perfiles
- Importancia de variables
- Matriz de confusión
- Curva ROC
- Función sigmoide
- Matriz de correlación
- Distribución por variable
- Métricas del modelo
- Pipeline del proyecto
