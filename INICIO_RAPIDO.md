# 🚀 Inicio Rápido - Clasificador de Perfiles de Estudio

Guía express para empezar en **5 minutos**.

---

## ⚡ Setup en 3 Pasos

### 1️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar la app

```bash
streamlit run streamlit_logistic_study_profile.py
```

### 3️⃣ Abrir en el navegador

La app se abrirá automáticamente en: `http://localhost:8501`

---

## 🎯 ¿Qué puedo hacer?

### Opción A: Probar la Encuesta (más rápido)

1. Ir al tab **"🎯 Encuesta Interactiva"**
2. Responder las 7 preguntas
3. Click en **"🔮 Predecir mi perfil"**
4. Ver tu resultado: Estratégico o Intuitivo

### Opción B: Explorar el Dataset

1. Ir al tab **"📊 Dataset & Modelo"**
2. Ver las métricas del modelo (Accuracy: ~87%)
3. Explorar la matriz de confusión

### Opción C: Análisis Avanzado

1. Ir al tab **"📈 Análisis"**
2. Ver importancia de variables
3. Explorar correlaciones

### Opción D: Exportar Recursos

1. Ir al tab **"💾 Exportar"**
2. Descargar modelo entrenado (.pkl)
3. Descargar dataset de ejemplo (CSV)
4. Descargar plantilla de encuesta

---

## 🖥️ Versión CLI (sin interfaz gráfica)

Si preferís terminal:

```bash
python clasificador_cli.py
```

Menú interactivo con opciones:
1. Entrenar modelo
2. Predecir perfil
3. Ver información del modelo

---

## 📓 Notebook Jupyter

Para análisis paso a paso:

```bash
jupyter notebook analisis_perfiles_estudio.ipynb
```

---

## 🎤 Preparar Presentación

### Generar gráficos para PowerPoint:

```bash
python generar_graficos_presentacion.py
```

Crea 9 gráficos profesionales en `graficos_presentacion/`

### Leer guía de presentación:

Abrir `GUIA_PRESENTACION.md` para:
- Estructura de slides (20-22)
- Guión completo
- Tips de diseño
- Preguntas frecuentes

---

## 📊 Ejemplo de Uso Rápido (CLI)

```bash
# Entrenar modelo
python clasificador_cli.py --entrenar

# Predecir tu perfil
python clasificador_cli.py --predecir
```

---

## 🤖 Usar Modelo en tu Código

```python
import joblib
import numpy as np

# Cargar modelo
modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
model = modelo_data['model']
scaler = modelo_data['scaler']

# Hacer predicción
# [planifica, usa_apps, estudia_solo, consulta_fuentes, 
#  prefiere_practica, procrastina, usa_resumenes]
respuestas = np.array([[1, 1, 0, 1, 1, 0, 1]])
respuestas_scaled = scaler.transform(respuestas)

prediccion = model.predict(respuestas_scaled)[0]
probabilidad = model.predict_proba(respuestas_scaled)[0][1]

print(f"Perfil: {'Estratégico' if prediccion == 1 else 'Intuitivo'}")
print(f"Probabilidad: {probabilidad:.2%}")
```

---

## 🆘 Problemas Comunes

### Error: "ModuleNotFoundError"
```bash
pip install --upgrade -r requirements.txt
```

### Error: "Port 8501 is already in use"
```bash
streamlit run streamlit_logistic_study_profile.py --server.port 8502
```

### La app no se abre
Abrir manualmente: `http://localhost:8501`

---

## 📚 Recursos Adicionales

| Archivo | Descripción |
|---------|-------------|
| `README.md` | Documentación completa |
| `GUIA_PRESENTACION.md` | Guía para presentar en clase |
| `streamlit_logistic_study_profile.py` | App principal |
| `clasificador_cli.py` | Versión terminal |
| `analisis_perfiles_estudio.ipynb` | Notebook completo |
| `generar_graficos_presentacion.py` | Generador de gráficos |

---

## 🎯 Flujo Recomendado para Clase

### Antes de la clase:
1. ✅ Instalar dependencias
2. ✅ Ejecutar app y verificar que funciona
3. ✅ Generar gráficos para presentación
4. ✅ Leer `GUIA_PRESENTACION.md`
5. ✅ Preparar PowerPoint con gráficos

### Durante la clase:
1. 📊 Presentar slides (teoría)
2. 🎬 Demo en vivo de la app
3. 🎯 Hacer que compañeros prueben la encuesta
4. 📈 Mostrar análisis de resultados
5. ❓ Responder preguntas

### Después de la clase:
1. 📤 Compartir código en GitHub
2. 📧 Enviar link de la app (si la deployás)
3. 📊 Compartir dataset y modelo

---

## 🚀 Deploy Online (Opcional)

### Streamlit Cloud (gratis):

1. Subir código a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar repositorio
4. Deploy automático

Tu app estará online en: `https://tu-usuario-clasificador.streamlit.app`

---

## 💡 Tips Finales

- 🎨 La app usa colores: **#667eea** (estratégico) y **#f56565** (intuitivo)
- 📊 El modelo tiene ~87% de accuracy
- 🔄 Podés cargar tu propio CSV en la barra lateral
- 💾 Exportá el modelo para usarlo en otros proyectos
- 📓 El notebook tiene explicaciones paso a paso
- 🎤 La guía de presentación tiene todo lo que necesitás

---

**¿Dudas? Lee el `README.md` completo o la `GUIA_PRESENTACION.md`**

**¡Éxito con tu proyecto! 🎉**
