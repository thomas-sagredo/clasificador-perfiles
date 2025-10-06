"""
📊 Generador de Gráficos para Presentación
Crea todas las imágenes necesarias para los slides de PowerPoint
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from pathlib import Path

# Configuración
OUTPUT_DIR = Path("graficos_presentacion")
OUTPUT_DIR.mkdir(exist_ok=True)

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("🎨 Generando gráficos para la presentación...\n")

# ============================================================================
# 1. GENERAR DATASET
# ============================================================================

def generar_dataset_simulado(n=300, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        'planifica': np.random.binomial(1, 0.5, n),
        'usa_apps': np.random.binomial(1, 0.45, n),
        'estudia_solo': np.random.binomial(1, 0.4, n),
        'consulta_fuentes': np.random.binomial(1, 0.6, n),
        'prefiere_practica': np.random.binomial(1, 0.55, n),
        'procrastina': np.random.binomial(1, 0.35, n),
        'usa_resumenes': np.random.binomial(1, 0.5, n)
    })
    
    coefs = np.array([1.2, 0.9, -0.3, 0.8, 0.4, -1.0, 0.7])
    intercept = -0.2
    logits = data.values.dot(coefs) + intercept
    probs = 1 / (1 + np.exp(-logits))
    labels = (probs > 0.5).astype(int)
    data['perfil'] = labels
    
    return data

df = generar_dataset_simulado(n=300)

# ============================================================================
# 2. ENTRENAR MODELO
# ============================================================================

X = df.drop(columns=['perfil'])
y = df['perfil']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("✅ Modelo entrenado\n")

# ============================================================================
# GRÁFICO 1: DISTRIBUCIÓN DE PERFILES (PIE CHART)
# ============================================================================

print("📊 Generando: 1_distribucion_perfiles.png")

estrategicos = df['perfil'].sum()
intuitivos = len(df) - estrategicos

fig = go.Figure(data=[go.Pie(
    labels=['💡 Intuitivo', '📚 Estratégico'],
    values=[intuitivos, estrategicos],
    hole=.4,
    marker_colors=['#f56565', '#667eea'],
    textinfo='label+percent+value',
    textfont_size=14
)])

fig.update_layout(
    title={
        'text': "Distribución de Perfiles en el Dataset",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    width=800,
    height=600,
    showlegend=True,
    legend=dict(font=dict(size=14))
)

fig.write_image(OUTPUT_DIR / "1_distribucion_perfiles.png")

# ============================================================================
# GRÁFICO 2: IMPORTANCIA DE VARIABLES (BARRAS HORIZONTALES)
# ============================================================================

print("📊 Generando: 2_importancia_variables.png")

coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': model.coef_[0]
}).sort_values('Coeficiente', ascending=True)

colors = ['#f56565' if c < 0 else '#667eea' for c in coef_df['Coeficiente']]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=coef_df['Coeficiente'],
    y=coef_df['Variable'],
    orientation='h',
    marker_color=colors,
    text=[f"{v:.2f}" for v in coef_df['Coeficiente']],
    textposition='outside',
    textfont=dict(size=12)
))

fig.update_layout(
    title={
        'text': "🎯 Importancia de Variables (Coeficientes del Modelo)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial Black'}
    },
    xaxis_title="Coeficiente",
    yaxis_title="Variable",
    width=900,
    height=600,
    showlegend=False,
    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
    font=dict(size=12)
)

fig.write_image(OUTPUT_DIR / "2_importancia_variables.png")

# ============================================================================
# GRÁFICO 3: MATRIZ DE CONFUSIÓN
# ============================================================================

print("📊 Generando: 3_matriz_confusion.png")

cm = confusion_matrix(y_test, y_pred)

fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Intuitivo', 'Estratégico'],
    y=['Intuitivo', 'Estratégico'],
    text=cm,
    texttemplate="%{text}",
    textfont={"size": 20},
    colorscale='Blues',
    showscale=True,
    colorbar=dict(title="Cantidad")
))

fig.update_layout(
    title={
        'text': "🎯 Matriz de Confusión (Test Set)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title="Predicción",
    yaxis_title="Real",
    width=700,
    height=600,
    font=dict(size=14)
)

fig.update_xaxis(side="bottom")

fig.write_image(OUTPUT_DIR / "3_matriz_confusion.png")

# ============================================================================
# GRÁFICO 4: CURVA ROC
# ============================================================================

print("📊 Generando: 4_curva_roc.png")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

fig = go.Figure()

# Curva ROC
fig.add_trace(go.Scatter(
    x=fpr, 
    y=tpr, 
    mode='lines',
    name=f'ROC (AUC = {roc_auc:.3f})',
    line=dict(color='#667eea', width=3),
    fill='tozeroy',
    fillcolor='rgba(102, 126, 234, 0.2)'
))

# Línea diagonal (random)
fig.add_trace(go.Scatter(
    x=[0, 1], 
    y=[0, 1], 
    mode='lines',
    name='Random (AUC = 0.500)',
    line=dict(color='gray', width=2, dash='dash')
))

fig.update_layout(
    title={
        'text': f"📈 Curva ROC (AUC = {roc_auc:.3f})",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title="False Positive Rate (FPR)",
    yaxis_title="True Positive Rate (TPR)",
    width=800,
    height=700,
    showlegend=True,
    legend=dict(x=0.6, y=0.1, font=dict(size=14)),
    xaxis=dict(range=[0, 1], gridcolor='lightgray'),
    yaxis=dict(range=[0, 1], gridcolor='lightgray'),
    plot_bgcolor='white'
)

fig.write_image(OUTPUT_DIR / "4_curva_roc.png")

# ============================================================================
# GRÁFICO 5: FUNCIÓN SIGMOIDE
# ============================================================================

print("📊 Generando: 5_funcion_sigmoide.png")

z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=z, 
    y=sigmoid, 
    mode='lines',
    line=dict(width=4, color='#667eea'),
    name='σ(z) = 1 / (1 + e^(-z))'
))

# Línea horizontal en 0.5
fig.add_hline(
    y=0.5, 
    line_dash="dash", 
    line_color="red", 
    line_width=2,
    annotation_text="Umbral = 0.5",
    annotation_position="right"
)

# Línea vertical en z=0
fig.add_vline(
    x=0, 
    line_dash="dot", 
    line_color="gray", 
    line_width=1
)

fig.update_layout(
    title={
        'text': "📐 Función Sigmoide",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title="z (logit)",
    yaxis_title="σ(z) - Probabilidad",
    width=900,
    height=600,
    showlegend=True,
    legend=dict(x=0.05, y=0.95, font=dict(size=14)),
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray', range=[0, 1]),
    plot_bgcolor='white'
)

fig.write_image(OUTPUT_DIR / "5_funcion_sigmoide.png")

# ============================================================================
# GRÁFICO 6: CORRELACIÓN ENTRE VARIABLES
# ============================================================================

print("📊 Generando: 6_matriz_correlacion.png")

corr_matrix = df.corr()

fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_matrix.values.round(2),
    texttemplate="%{text}",
    textfont={"size": 11},
    colorbar=dict(title="Correlación")
))

fig.update_layout(
    title={
        'text': "🔗 Matriz de Correlación entre Variables",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial Black'}
    },
    width=900,
    height=800,
    font=dict(size=12)
)

fig.write_image(OUTPUT_DIR / "6_matriz_correlacion.png")

# ============================================================================
# GRÁFICO 7: DISTRIBUCIÓN POR VARIABLE Y PERFIL
# ============================================================================

print("📊 Generando: 7_distribucion_variables.png")

variables = ['planifica', 'usa_apps', 'estudia_solo', 'consulta_fuentes', 
             'prefiere_practica', 'procrastina', 'usa_resumenes']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('📊 Distribución de Variables por Perfil', fontsize=18, fontweight='bold')

for idx, var in enumerate(variables):
    ax = axes[idx // 4, idx % 4]
    
    prop_estrategico = df[df['perfil'] == 1][var].mean()
    prop_intuitivo = df[df['perfil'] == 0][var].mean()
    
    bars = ax.bar(['Intuitivo', 'Estratégico'], [prop_intuitivo, prop_estrategico], 
                  color=['#f56565', '#667eea'], edgecolor='black', linewidth=1.5)
    
    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title(var, fontweight='bold', fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proporción', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "7_distribucion_variables.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# GRÁFICO 8: MÉTRICAS DEL MODELO
# ============================================================================

print("📊 Generando: 8_metricas_modelo.png")

from sklearn.metrics import precision_score, recall_score, f1_score

accuracy = (y_pred == y_test).mean()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
valores = [accuracy, precision, recall, f1, roc_auc]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=metricas,
    y=valores,
    text=[f"{v:.1%}" for v in valores],
    textposition='outside',
    marker_color='#667eea',
    marker_line_color='black',
    marker_line_width=2
))

fig.update_layout(
    title={
        'text': "📊 Métricas de Evaluación del Modelo",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    yaxis_title="Valor",
    width=900,
    height=600,
    yaxis=dict(range=[0, 1], tickformat='.0%'),
    showlegend=False,
    font=dict(size=14)
)

fig.write_image(OUTPUT_DIR / "8_metricas_modelo.png")

# ============================================================================
# GRÁFICO 9: PIPELINE DEL PROYECTO
# ============================================================================

print("📊 Generando: 9_pipeline.png")

fig = go.Figure()

# Nodos del pipeline
steps = [
    "1. Recolección\nde Datos",
    "2. Análisis\nExploratorio",
    "3. Preprocesamiento\n(Scaling)",
    "4. Entrenamiento\n(Logistic Reg.)",
    "5. Evaluación\n(Métricas)",
    "6. Deployment\n(Streamlit)"
]

x_positions = list(range(len(steps)))
y_positions = [0] * len(steps)

# Agregar nodos
fig.add_trace(go.Scatter(
    x=x_positions,
    y=y_positions,
    mode='markers+text',
    marker=dict(size=80, color='#667eea', line=dict(width=3, color='black')),
    text=steps,
    textposition='middle center',
    textfont=dict(size=10, color='white', family='Arial Black'),
    showlegend=False
))

# Agregar flechas
for i in range(len(steps) - 1):
    fig.add_annotation(
        x=x_positions[i+1],
        y=0,
        ax=x_positions[i],
        ay=0,
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='black'
    )

fig.update_layout(
    title={
        'text': "🔄 Pipeline del Proyecto de Machine Learning",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    width=1200,
    height=400,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1, 1]),
    plot_bgcolor='white'
)

fig.write_image(OUTPUT_DIR / "9_pipeline.png")

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*60)
print("✅ GRÁFICOS GENERADOS EXITOSAMENTE".center(60))
print("="*60)
print(f"\n📁 Ubicación: {OUTPUT_DIR.absolute()}\n")
print("📊 Archivos creados:")
print("   1. 1_distribucion_perfiles.png")
print("   2. 2_importancia_variables.png")
print("   3. 3_matriz_confusion.png")
print("   4. 4_curva_roc.png")
print("   5. 5_funcion_sigmoide.png")
print("   6. 6_matriz_correlacion.png")
print("   7. 7_distribucion_variables.png")
print("   8. 8_metricas_modelo.png")
print("   9. 9_pipeline.png")
print("\n💡 Estos gráficos están listos para insertar en PowerPoint")
print("="*60 + "\n")
