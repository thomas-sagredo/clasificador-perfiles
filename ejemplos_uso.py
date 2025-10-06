"""
📚 Ejemplos de Uso del Modelo
Diferentes formas de usar el clasificador de perfiles de estudio
"""

import joblib
import numpy as np
import pandas as pd

# ============================================================================
# EJEMPLO 1: Cargar y usar modelo entrenado
# ============================================================================

def ejemplo_1_uso_basico():
    """Uso básico del modelo guardado"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Uso Básico del Modelo")
    print("="*60 + "\n")
    
    # Cargar modelo
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    columns = modelo_data['columns']
    
    print(f"✅ Modelo cargado")
    print(f"   Accuracy: {modelo_data['accuracy']:.2%}")
    print(f"   Variables: {columns}\n")
    
    # Hacer predicción
    # Formato: [planifica, usa_apps, estudia_solo, consulta_fuentes, 
    #           prefiere_practica, procrastina, usa_resumenes]
    respuestas = np.array([[1, 1, 0, 1, 1, 0, 1]])
    respuestas_scaled = scaler.transform(respuestas)
    
    prediccion = model.predict(respuestas_scaled)[0]
    probabilidad = model.predict_proba(respuestas_scaled)[0][1]
    
    print("📝 Respuestas:")
    for col, val in zip(columns, respuestas[0]):
        print(f"   {col}: {'Sí' if val == 1 else 'No'}")
    
    print(f"\n🎯 Resultado:")
    print(f"   Perfil: {'📚 Estratégico' if prediccion == 1 else '💡 Intuitivo'}")
    print(f"   Confianza: {probabilidad:.1%}")


# ============================================================================
# EJEMPLO 2: Predecir múltiples estudiantes
# ============================================================================

def ejemplo_2_batch_prediction():
    """Predecir perfiles de múltiples estudiantes a la vez"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Predicción en Batch")
    print("="*60 + "\n")
    
    # Cargar modelo
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    
    # Dataset con 5 estudiantes
    estudiantes = pd.DataFrame({
        'nombre': ['Ana', 'Bruno', 'Carla', 'Diego', 'Elena'],
        'planifica': [1, 0, 1, 0, 1],
        'usa_apps': [1, 0, 1, 0, 0],
        'estudia_solo': [0, 1, 0, 1, 1],
        'consulta_fuentes': [1, 0, 1, 1, 0],
        'prefiere_practica': [1, 1, 0, 1, 1],
        'procrastina': [0, 1, 0, 1, 1],
        'usa_resumenes': [1, 0, 1, 0, 0]
    })
    
    # Separar features
    X = estudiantes.drop(columns=['nombre'])
    X_scaled = scaler.transform(X)
    
    # Predecir
    predicciones = model.predict(X_scaled)
    probabilidades = model.predict_proba(X_scaled)[:, 1]
    
    # Agregar resultados
    estudiantes['perfil'] = ['Estratégico' if p == 1 else 'Intuitivo' for p in predicciones]
    estudiantes['confianza'] = [f"{p:.1%}" for p in probabilidades]
    
    print("📊 Resultados:")
    print(estudiantes[['nombre', 'perfil', 'confianza']].to_string(index=False))


# ============================================================================
# EJEMPLO 3: Análisis de sensibilidad
# ============================================================================

def ejemplo_3_analisis_sensibilidad():
    """Ver cómo cambia la predicción al modificar una variable"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Análisis de Sensibilidad")
    print("="*60 + "\n")
    
    # Cargar modelo
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    columns = modelo_data['columns']
    
    # Perfil base
    base = np.array([1, 1, 0, 1, 1, 0, 1])
    base_scaled = scaler.transform([base])
    prob_base = model.predict_proba(base_scaled)[0][1]
    
    print(f"🎯 Perfil Base:")
    print(f"   Probabilidad Estratégico: {prob_base:.1%}\n")
    
    print("📊 Impacto de cambiar cada variable:\n")
    
    for i, col in enumerate(columns):
        # Cambiar solo esa variable
        modificado = base.copy()
        modificado[i] = 1 - modificado[i]  # Invertir
        
        modificado_scaled = scaler.transform([modificado])
        prob_modificado = model.predict_proba(modificado_scaled)[0][1]
        
        cambio = prob_modificado - prob_base
        signo = "⬆️" if cambio > 0 else "⬇️"
        
        print(f"   {col:<20} {signo} {cambio:+.1%}")


# ============================================================================
# EJEMPLO 4: Crear función de recomendación
# ============================================================================

def ejemplo_4_recomendaciones():
    """Generar recomendaciones según el perfil detectado"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Sistema de Recomendaciones")
    print("="*60 + "\n")
    
    # Cargar modelo
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    
    # Ejemplo de estudiante
    respuestas = np.array([[0, 0, 1, 0, 1, 1, 0]])
    respuestas_scaled = scaler.transform(respuestas)
    
    prediccion = model.predict(respuestas_scaled)[0]
    probabilidad = model.predict_proba(respuestas_scaled)[0][1]
    
    print(f"🎯 Perfil detectado: {'Estratégico' if prediccion == 1 else 'Intuitivo'}")
    print(f"   Confianza: {probabilidad:.1%}\n")
    
    # Recomendaciones personalizadas
    print("💡 Recomendaciones:\n")
    
    if prediccion == 1:  # Estratégico
        print("   ✅ Seguí usando tus estrategias de planificación")
        print("   ✅ Considerá compartir tus métodos con otros")
        print("   📚 Recursos sugeridos:")
        print("      - Notion para organización")
        print("      - Pomodoro Technique")
        print("      - Mind mapping tools")
    else:  # Intuitivo
        print("   💡 Probá incorporar algo de planificación gradualmente")
        print("   💡 Usá apps simples para organizarte (Google Calendar)")
        print("   📚 Recursos sugeridos:")
        print("      - Técnicas de estudio activo")
        print("      - Flashcards (Anki)")
        print("      - Grupos de estudio")
    
    # Recomendaciones específicas según respuestas
    if respuestas[0][5] == 1:  # Procrastina
        print("\n   ⚠️ Detectamos procrastinación:")
        print("      - Probá la técnica de 'comer la rana'")
        print("      - Dividí tareas grandes en pequeñas")
        print("      - Usá timers (25 min de trabajo)")


# ============================================================================
# EJEMPLO 5: Integración con CSV
# ============================================================================

def ejemplo_5_procesar_csv():
    """Procesar un CSV con respuestas de múltiples estudiantes"""
    print("\n" + "="*60)
    print("EJEMPLO 5: Procesar CSV de Encuestas")
    print("="*60 + "\n")
    
    # Crear CSV de ejemplo
    ejemplo_csv = pd.DataFrame({
        'nombre': ['Estudiante1', 'Estudiante2', 'Estudiante3'],
        'planifica': [1, 0, 1],
        'usa_apps': [1, 0, 0],
        'estudia_solo': [0, 1, 1],
        'consulta_fuentes': [1, 1, 0],
        'prefiere_practica': [1, 1, 1],
        'procrastina': [0, 1, 0],
        'usa_resumenes': [1, 0, 1]
    })
    
    # Guardar CSV temporal
    ejemplo_csv.to_csv('ejemplo_respuestas.csv', index=False)
    print("✅ CSV de ejemplo creado: ejemplo_respuestas.csv\n")
    
    # Cargar modelo
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    
    # Leer CSV
    df = pd.read_csv('ejemplo_respuestas.csv')
    
    # Predecir
    X = df.drop(columns=['nombre'])
    X_scaled = scaler.transform(X)
    
    predicciones = model.predict(X_scaled)
    probabilidades = model.predict_proba(X_scaled)[:, 1]
    
    # Agregar resultados
    df['perfil'] = ['Estratégico' if p == 1 else 'Intuitivo' for p in predicciones]
    df['probabilidad_estrategico'] = probabilidades
    
    # Guardar resultados
    df.to_csv('resultados_clasificacion.csv', index=False)
    
    print("📊 Resultados guardados en: resultados_clasificacion.csv\n")
    print(df[['nombre', 'perfil', 'probabilidad_estrategico']].to_string(index=False))


# ============================================================================
# EJEMPLO 6: Validación cruzada personalizada
# ============================================================================

def ejemplo_6_validacion_custom():
    """Validar el modelo con diferentes métricas"""
    print("\n" + "="*60)
    print("EJEMPLO 6: Validación Personalizada")
    print("="*60 + "\n")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
    
    # Cargar dataset
    df = pd.read_csv('dataset_perfiles_estudio.csv')
    X = df.drop(columns=['perfil'])
    y = df['perfil']
    
    # Cargar modelo y scaler
    modelo_data = joblib.load('modelo_perfiles_estudio.pkl')
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    
    # Escalar
    X_scaled = scaler.transform(X)
    
    # Diferentes métricas
    metricas = {
        'Accuracy': 'accuracy',
        'Precision': make_scorer(precision_score),
        'Recall': make_scorer(recall_score),
        'F1-Score': make_scorer(f1_score)
    }
    
    print("📊 Validación Cruzada (5-fold):\n")
    
    for nombre, metrica in metricas.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring=metrica)
        print(f"   {nombre:<12} {scores.mean():.2%} ± {scores.std():.2%}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecutar todos los ejemplos"""
    print("\n" + "="*60)
    print("📚 EJEMPLOS DE USO DEL CLASIFICADOR".center(60))
    print("="*60)
    
    try:
        # Verificar que existe el modelo
        import os
        if not os.path.exists('modelo_perfiles_estudio.pkl'):
            print("\n❌ Error: No se encontró 'modelo_perfiles_estudio.pkl'")
            print("   Ejecutá primero la app o el CLI para entrenar el modelo.")
            return
        
        # Ejecutar ejemplos
        ejemplo_1_uso_basico()
        ejemplo_2_batch_prediction()
        ejemplo_3_analisis_sensibilidad()
        ejemplo_4_recomendaciones()
        ejemplo_5_procesar_csv()
        ejemplo_6_validacion_custom()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS EJEMPLOS EJECUTADOS EXITOSAMENTE".center(60))
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Asegurate de haber entrenado el modelo primero.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")


if __name__ == '__main__':
    main()
