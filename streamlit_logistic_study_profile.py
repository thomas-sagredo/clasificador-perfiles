# streamlit_logistic_study_profile.py
# Mini-app: Encuesta + Regresión Logística para clasificar perfiles de estudio
# Requisitos: streamlit, scikit-learn, pandas, numpy, joblib

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import io
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="🧪 Clasificador de Perfiles de Estudio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helpers ----------
@st.cache_data
def generar_dataset_simulado(n=300, seed=42):
    """Genera dataset simulado con correlaciones realistas entre variables"""
    np.random.seed(seed)
    # Preguntas binarias (0/1)
    data = pd.DataFrame({
        'planifica': np.random.binomial(1, 0.5, n),
        'usa_apps': np.random.binomial(1, 0.45, n),
        'estudia_solo': np.random.binomial(1, 0.4, n),
        'consulta_fuentes': np.random.binomial(1, 0.6, n),
        'prefiere_practica': np.random.binomial(1, 0.55, n),
        'procrastina': np.random.binomial(1, 0.35, n),
        'usa_resumenes': np.random.binomial(1, 0.5, n)
    })
    # Construimos una "probabilidad verdadera" lineal para generar la etiqueta
    # Variables que aumentan probabilidad de ser 'estratégico': planifica, usa_apps, usa_resumenes, consulta_fuentes
    # Variables que disminuyen: procrastina
    coefs = np.array([1.2, 0.9, -0.3, 0.8, 0.4, -1.0, 0.7])
    intercept = -0.2
    logits = data.values.dot(coefs) + intercept
    probs = 1 / (1 + np.exp(-logits))
    labels = (probs > 0.5).astype(int)
    data['perfil'] = labels
    return data

@st.cache_data
def entrenar_modelo(df, target_col='perfil'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'X_columns': X.columns.tolist()
    }

# ---------- App ----------
st.title("🧪 Clasificador de Perfiles de Estudio")
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: white; margin: 0;'>📊 Mini Encuesta — Regresión Logística</h3>
    <p style='color: white; margin: 5px 0 0 0;'>Clasificamos perfiles de estudio en <strong>Estratégicos</strong> vs <strong>Intuitivos</strong></p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("---")
use_uploaded = st.sidebar.checkbox("📁 Cargar CSV propio para entrenar", value=False)

if use_uploaded:
    uploaded = st.sidebar.file_uploader(
        "Subí un CSV con las columnas:",
        help="Columnas requeridas: planifica, usa_apps, estudia_solo, consulta_fuentes, prefiere_practica, procrastina, usa_resumenes, perfil (opcional)"
    )
    if uploaded is not None:
        try:
            df_user = pd.read_csv(uploaded)
            st.sidebar.success("✅ Archivo cargado correctamente")
        except Exception as e:
            st.sidebar.error(f"❌ Error al leer CSV: {str(e)}")
            df_user = None
    else:
        df_user = None
else:
    df_user = None

# Configuración adicional en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Personalización")
show_stats = st.sidebar.checkbox("📈 Mostrar estadísticas detalladas", value=True)
show_coef_chart = st.sidebar.checkbox("📊 Mostrar importancia de variables", value=True)

# Generar dataset simulado si no se subió uno
if df_user is None:
    st.info("💡 Usando dataset simulado (podés subir tu propio CSV en la barra lateral)")
    df = generar_dataset_simulado(n=300)
else:
    df = df_user.copy()
    if 'perfil' not in df.columns:
        st.warning("⚠️ El CSV subido no tiene columna 'perfil'. El modelo se entrenará con el dataset simulado.")

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset & Modelo", "🎯 Encuesta Interactiva", "📈 Análisis", "💾 Exportar"])

# Entrenamiento del modelo
if 'perfil' in df.columns:
    with st.spinner('🔄 Entrenando modelo...'):
        result = entrenar_modelo(df)
    model = result['model']
    scaler = result['scaler']
    cols = result['X_columns']
    dataset_usado = df
else:
    st.info("ℹ️ No hay etiqueta 'perfil' en el dataset. Entrenando con dataset simulado.")
    df_sim = generar_dataset_simulado(n=300)
    result = entrenar_modelo(df_sim)
    model = result['model']
    scaler = result['scaler']
    cols = result['X_columns']
    dataset_usado = df_sim

# TAB 1: Dataset & Modelo
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Accuracy", f"{result['accuracy']:.2%}")
    with col2:
        estrategicos = dataset_usado['perfil'].sum()
        st.metric("📚 Estratégicos", f"{estrategicos} ({estrategicos/len(dataset_usado):.1%})")
    with col3:
        intuitivos = len(dataset_usado) - estrategicos
        st.metric("💡 Intuitivos", f"{intuitivos} ({intuitivos/len(dataset_usado):.1%})")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📋 Vista previa del dataset")
        st.dataframe(dataset_usado.head(10), use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Matriz de Confusión")
        cm = result['confusion_matrix']
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Intuitivo', 'Estratégico'],
            y=['Intuitivo', 'Estratégico'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(
            title="Predicción vs Real",
            xaxis_title="Predicción",
            yaxis_title="Real",
            height=300
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    if show_stats:
        st.markdown("---")
        st.subheader("📊 Reporte de Clasificación")
        report_df = pd.DataFrame(result['report']).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# TAB 2: Encuesta Interactiva
with tab2:
    st.subheader("🎯 Descubrí tu perfil de estudio")
    st.write("Completá las preguntas y el modelo clasificará tu perfil con la probabilidad correspondiente.")
    
    st.markdown("---")
    
    # Preguntas con emojis y mejor diseño
    preguntas = {
        'planifica': ('📅', '¿Planificás tu semana de estudio con anticipación?'),
        'usa_apps': ('📱', '¿Usás herramientas digitales para organizarte (calendario, apps)?'),
        'estudia_solo': ('🧑', '¿Preferís estudiar solo/a antes que en grupo?'),
        'consulta_fuentes': ('🌐', '¿Consultás fuentes externas (videos, IA, foros) para entender temas?'),
        'prefiere_practica': ('✍️', '¿Sentís que aprendés más resolviendo ejercicios que leyendo teoría?'),
        'procrastina': ('⏰', '¿Sos de dejar todo para último momento?'),
        'usa_resumenes': ('📝', '¿Te resulta útil repasar con resúmenes o mapas mentales?')
    }
    
    inputs = {}
    col_left, col_right = st.columns(2)
    
    items = list(preguntas.items())
    for i, (key, (emoji, pregunta)) in enumerate(items):
        col = col_left if i % 2 == 0 else col_right
        with col:
            inputs[key] = st.radio(f"{emoji} {pregunta}", ('No', 'Sí'), key=key)
    
    st.markdown("---")
    
    # Convertir respuestas a formato numérico
    def resp_to_num(val):
        return 1 if val == 'Sí' else 0
    
    x_input = np.array([[resp_to_num(inputs[c]) for c in cols]])
    
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button('🔮 Predecir mi perfil', type="primary", use_container_width=True)
    
    if predict_btn:
        x_s = scaler.transform(x_input)
        prob = model.predict_proba(x_s)[0][1]
        pred = model.predict(x_s)[0]
        label = '📚 Estratégico' if pred == 1 else '💡 Intuitivo'
        
        st.markdown("---")
        st.subheader("🎯 Resultado de tu perfil")
        
        # Métricas visuales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Perfil Detectado", label)
        with col2:
            st.metric("Confianza Estratégico", f"{prob*100:.1f}%")
        with col3:
            st.metric("Confianza Intuitivo", f"{(1-prob)*100:.1f}%")
        
        # Gauge chart de probabilidad
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilidad de ser Estratégico", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "#667eea"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffd6d6'},
                    {'range': [40, 60], 'color': '#fff4d6'},
                    {'range': [60, 100], 'color': '#d6f5d6'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Interpretación
        st.markdown("### 💬 Interpretación")
        if prob > 0.8:
            st.success('🌟 **Muy probable que seas estratégico** — Planificás, organizás y usás estrategias de estudio estructuradas.')
        elif prob > 0.6:
            st.info('📚 **Probablemente estratégico** — Tenés tendencia a organizar tu estudio de forma sistemática.')
        elif prob > 0.4:
            st.warning('🤔 **Perfil mixto** — Combinás elementos de ambos perfiles. Podrías beneficiarte de estrategias más concretas.')
        else:
            st.info('💡 **Más probable que seas intuitivo** — Aprendés mejor por práctica y respuestas rápidas, sin mucha planificación previa.')
        
        # Mostrar contribución de cada respuesta
        if show_coef_chart:
            st.markdown("---")
            st.subheader("📊 ¿Qué influyó en tu resultado?")
            
            coefs = pd.Series(result['model'].coef_[0], index=cols)
            respuestas_usuario = pd.Series([resp_to_num(inputs[c]) for c in cols], index=cols)
            contribuciones = coefs * respuestas_usuario
            
            fig_contrib = go.Figure()
            colors = ['#667eea' if c > 0 else '#f56565' for c in contribuciones]
            fig_contrib.add_trace(go.Bar(
                x=contribuciones.sort_values().values,
                y=contribuciones.sort_values().index,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.2f}" for v in contribuciones.sort_values().values],
                textposition='outside'
            ))
            fig_contrib.update_layout(
                title="Contribución de cada respuesta a tu perfil",
                xaxis_title="Impacto (+ Estratégico / - Intuitivo)",
                yaxis_title="Variable",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_contrib, use_container_width=True)

# TAB 3: Análisis
with tab3:
    st.subheader("📈 Análisis del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Importancia de Variables")
        coefs = pd.Series(result['model'].coef_[0], index=cols)
        fig_coef = go.Figure()
        colors = ['#667eea' if c > 0 else '#f56565' for c in coefs.sort_values()]
        fig_coef.add_trace(go.Bar(
            x=coefs.sort_values().values,
            y=coefs.sort_values().index,
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in coefs.sort_values().values],
            textposition='outside'
        ))
        fig_coef.update_layout(
            xaxis_title="Coeficiente",
            yaxis_title="Variable",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption("Coeficientes positivos → aumentan probabilidad de ser Estratégico")
    
    with col2:
        st.markdown("### 📊 Distribución de Perfiles")
        perfil_counts = dataset_usado['perfil'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Intuitivo', 'Estratégico'],
            values=[perfil_counts.get(0, 0), perfil_counts.get(1, 0)],
            hole=.4,
            marker_colors=['#f56565', '#667eea']
        )])
        fig_pie.update_layout(
            title="Distribución en el dataset",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Análisis de correlaciones
    st.markdown("### 🔗 Correlaciones entre Variables")
    corr_matrix = dataset_usado.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig_corr.update_layout(
        title="Matriz de Correlación",
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Estadísticas por perfil
    st.markdown("---")
    st.markdown("### 📋 Estadísticas por Perfil")
    stats_por_perfil = dataset_usado.groupby('perfil').mean()
    stats_por_perfil.index = ['Intuitivo (0)', 'Estratégico (1)']
    st.dataframe(stats_por_perfil.style.background_gradient(cmap='RdYlGn', axis=1).format("{:.2f}"), use_container_width=True)

# TAB 4: Exportar
with tab4:
    st.subheader("💾 Exportar Recursos")
    st.write("Descargá el modelo entrenado y datasets para usar en clase o compartir con compañeros.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Modelo Entrenado")
        st.write("Descargá el modelo (.pkl) para usarlo en otros scripts o notebooks.")
        buf = io.BytesIO()
        joblib.dump({'model': model, 'scaler': scaler, 'columns': cols}, buf)
        buf.seek(0)
        st.download_button(
            label='📥 Descargar modelo (.pkl)',
            data=buf,
            file_name='modelo_perfiles_estudio.pkl',
            mime='application/octet-stream',
            use_container_width=True
        )
        st.info("El archivo incluye: modelo, scaler y nombres de columnas")
    
    with col2:
        st.markdown("### 📊 Dataset Simulado")
        st.write("Descargá el dataset de ejemplo para distribuir a los compañeros.")
        csv = generar_dataset_simulado(n=300).to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Descargar dataset (CSV)',
            data=csv,
            file_name='dataset_simulado_perfiles.csv',
            mime='text/csv',
            use_container_width=True
        )
        st.info("300 registros simulados con etiquetas")
    
    st.markdown("---")
    
    # Plantilla de encuesta
    st.markdown("### 📋 Plantilla de Encuesta")
    st.write("Copiá este formato para recolectar respuestas de tus compañeros:")
    
    plantilla = """Nombre,planifica,usa_apps,estudia_solo,consulta_fuentes,prefiere_practica,procrastina,usa_resumenes
Ejemplo1,1,1,0,1,1,0,1
Ejemplo2,0,0,1,0,1,1,0
"""
    st.code(plantilla, language="csv")
    st.download_button(
        label='📥 Descargar plantilla CSV',
        data=plantilla.encode('utf-8'),
        file_name='plantilla_encuesta.csv',
        mime='text/csv'
    )
    
    st.markdown("---")

st.markdown('---')
st.caption('🧪 Código creado para trabajo practico de inteligencia artificial')
