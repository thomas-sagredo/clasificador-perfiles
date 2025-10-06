#!/usr/bin/env python3
"""
🧪 Clasificador de Perfiles de Estudio - Versión CLI
Versión simplificada sin interfaz gráfica para ejecutar en terminal.

Uso:
    python clasificador_cli.py                    # Modo interactivo
    python clasificador_cli.py --entrenar         # Entrenar nuevo modelo
    python clasificador_cli.py --predecir         # Predecir con modelo existente
    python clasificador_cli.py --csv datos.csv    # Entrenar con CSV propio
"""

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_dataset_simulado(n=300, seed=42):
    """Genera dataset simulado con correlaciones realistas."""
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


def entrenar_modelo(df):
    """Entrena el modelo de regresión logística."""
    X = df.drop(columns=['perfil'])
    y = df['perfil']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': acc,
        'confusion_matrix': cm,
        'report': report,
        'columns': list(X.columns)
    }


def guardar_modelo(resultado, filename='modelo_perfiles_estudio.pkl'):
    """Guarda el modelo entrenado."""
    modelo_data = {
        'model': resultado['model'],
        'scaler': resultado['scaler'],
        'columns': resultado['columns'],
        'accuracy': resultado['accuracy']
    }
    joblib.dump(modelo_data, filename)
    return filename


def cargar_modelo(filename='modelo_perfiles_estudio.pkl'):
    """Carga un modelo previamente entrenado."""
    if not Path(filename).exists():
        return None
    return joblib.load(filename)


def predecir_perfil(model, scaler, respuestas):
    """
    Realiza predicción con el modelo.
    
    Args:
        model: Modelo entrenado
        scaler: Scaler para normalizar
        respuestas: Array con 7 valores binarios [0 o 1]
    
    Returns:
        tuple: (prediccion, probabilidad)
    """
    respuestas_array = np.array([respuestas])
    respuestas_scaled = scaler.transform(respuestas_array)
    
    prediccion = model.predict(respuestas_scaled)[0]
    probabilidad = model.predict_proba(respuestas_scaled)[0][1]
    
    return prediccion, probabilidad


# ============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS
# ============================================================================

def mostrar_banner():
    """Muestra el banner de bienvenida."""
    print("\n" + "="*60)
    print("🧪 CLASIFICADOR DE PERFILES DE ESTUDIO".center(60))
    print("Regresión Logística - Versión CLI".center(60))
    print("="*60 + "\n")


def modo_entrenar(csv_path=None):
    """Modo de entrenamiento del modelo."""
    print("📊 MODO ENTRENAMIENTO\n")
    
    # Cargar o generar datos
    if csv_path and Path(csv_path).exists():
        print(f"📁 Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'perfil' not in df.columns:
            print("❌ Error: El CSV debe tener una columna 'perfil' con valores 0/1")
            return
    else:
        if csv_path:
            print(f"⚠️  Archivo no encontrado: {csv_path}")
        print("💡 Generando dataset simulado (300 registros)...")
        df = generar_dataset_simulado(n=300)
    
    print(f"✅ Dataset cargado: {len(df)} registros")
    print(f"   📚 Estratégicos: {df['perfil'].sum()} ({df['perfil'].mean():.1%})")
    print(f"   💡 Intuitivos: {(1-df['perfil']).sum()} ({(1-df['perfil']).mean():.1%})")
    
    # Entrenar
    print("\n🔄 Entrenando modelo...")
    resultado = entrenar_modelo(df)
    
    # Mostrar resultados
    print(f"\n✅ Modelo entrenado exitosamente!")
    print(f"   🎯 Accuracy: {resultado['accuracy']:.2%}")
    
    print("\n📊 Matriz de Confusión:")
    cm = resultado['confusion_matrix']
    print(f"   {'':>15} Pred: Intuitivo  Pred: Estratégico")
    print(f"   {'Real: Intuitivo':<15} {cm[0,0]:^15} {cm[0,1]:^18}")
    print(f"   {'Real: Estratégico':<15} {cm[1,0]:^15} {cm[1,1]:^18}")
    
    print("\n🎯 Importancia de Variables:")
    coefs = pd.DataFrame({
        'Variable': resultado['columns'],
        'Coeficiente': resultado['model'].coef_[0]
    }).sort_values('Coeficiente', ascending=False)
    
    for _, row in coefs.iterrows():
        signo = "⬆️" if row['Coeficiente'] > 0 else "⬇️"
        print(f"   {signo} {row['Variable']:<20} {row['Coeficiente']:>6.3f}")
    
    # Guardar modelo
    filename = guardar_modelo(resultado)
    print(f"\n💾 Modelo guardado en: {filename}")
    
    # Guardar dataset
    df.to_csv('dataset_perfiles_estudio.csv', index=False)
    print(f"💾 Dataset guardado en: dataset_perfiles_estudio.csv")


def modo_predecir():
    """Modo de predicción con modelo existente."""
    print("🔮 MODO PREDICCIÓN\n")
    
    # Cargar modelo
    modelo_data = cargar_modelo()
    if modelo_data is None:
        print("❌ Error: No se encontró modelo entrenado.")
        print("   Ejecutá primero: python clasificador_cli.py --entrenar")
        return
    
    model = modelo_data['model']
    scaler = modelo_data['scaler']
    columns = modelo_data['columns']
    
    print(f"✅ Modelo cargado (Accuracy: {modelo_data['accuracy']:.2%})")
    print("\n" + "="*60)
    print("📝 ENCUESTA DE PERFIL DE ESTUDIO")
    print("="*60)
    print("\nRespondé las siguientes preguntas con 0 (No) o 1 (Sí):\n")
    
    # Preguntas
    preguntas = {
        'planifica': '📅 ¿Planificás tu semana de estudio con anticipación?',
        'usa_apps': '📱 ¿Usás herramientas digitales para organizarte?',
        'estudia_solo': '🧑 ¿Preferís estudiar solo/a antes que en grupo?',
        'consulta_fuentes': '🌐 ¿Consultás fuentes externas (videos, IA, foros)?',
        'prefiere_practica': '✍️  ¿Aprendés más resolviendo ejercicios que leyendo?',
        'procrastina': '⏰ ¿Sos de dejar todo para último momento?',
        'usa_resumenes': '📝 ¿Te resulta útil repasar con resúmenes o mapas?'
    }
    
    respuestas = []
    for col in columns:
        while True:
            try:
                resp = input(f"{preguntas[col]} (0/1): ").strip()
                if resp in ['0', '1']:
                    respuestas.append(int(resp))
                    break
                else:
                    print("   ⚠️  Por favor ingresá 0 o 1")
            except KeyboardInterrupt:
                print("\n\n❌ Predicción cancelada")
                return
    
    # Predecir
    prediccion, probabilidad = predecir_perfil(model, scaler, respuestas)
    
    # Mostrar resultado
    print("\n" + "="*60)
    print("🎯 RESULTADO DE LA PREDICCIÓN".center(60))
    print("="*60)
    
    perfil = "📚 Estratégico" if prediccion == 1 else "💡 Intuitivo"
    print(f"\n   Perfil detectado: {perfil}")
    print(f"\n   Probabilidades:")
    print(f"      Estratégico: {probabilidad:>5.1%}")
    print(f"      Intuitivo:   {(1-probabilidad):>5.1%}")
    
    # Interpretación
    print(f"\n💬 Interpretación:")
    if probabilidad > 0.8:
        print("   🌟 Muy probable que seas estratégico")
        print("   Planificás y usás estrategias de estudio estructuradas.")
    elif probabilidad > 0.6:
        print("   📚 Probablemente estratégico")
        print("   Tenés tendencia a organizar tu estudio de forma sistemática.")
    elif probabilidad > 0.4:
        print("   🤔 Perfil mixto")
        print("   Combinás elementos de ambos perfiles.")
    else:
        print("   💡 Más probable que seas intuitivo")
        print("   Aprendés mejor por práctica sin mucha planificación previa.")
    
    print("\n" + "="*60 + "\n")


def modo_interactivo():
    """Modo interactivo con menú."""
    mostrar_banner()
    
    while True:
        print("\n📋 MENÚ PRINCIPAL")
        print("="*60)
        print("1. 🔄 Entrenar nuevo modelo (dataset simulado)")
        print("2. 📁 Entrenar con CSV propio")
        print("3. 🔮 Predecir mi perfil")
        print("4. 📊 Ver información del modelo actual")
        print("5. ❌ Salir")
        print("="*60)
        
        opcion = input("\nSeleccioná una opción (1-5): ").strip()
        
        if opcion == '1':
            print()
            modo_entrenar()
        
        elif opcion == '2':
            csv_path = input("\nIngresá la ruta del archivo CSV: ").strip()
            print()
            modo_entrenar(csv_path)
        
        elif opcion == '3':
            print()
            modo_predecir()
        
        elif opcion == '4':
            print("\n📊 INFORMACIÓN DEL MODELO\n")
            modelo_data = cargar_modelo()
            if modelo_data:
                print(f"✅ Modelo encontrado")
                print(f"   🎯 Accuracy: {modelo_data['accuracy']:.2%}")
                print(f"   📊 Variables: {', '.join(modelo_data['columns'])}")
            else:
                print("❌ No hay modelo entrenado")
        
        elif opcion == '5':
            print("\n👋 ¡Hasta luego!\n")
            break
        
        else:
            print("\n⚠️  Opción inválida. Por favor seleccioná 1-5.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🧪 Clasificador de Perfiles de Estudio - CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python clasificador_cli.py                      # Modo interactivo
  python clasificador_cli.py --entrenar           # Entrenar con datos simulados
  python clasificador_cli.py --predecir           # Predecir perfil
  python clasificador_cli.py --csv datos.csv      # Entrenar con CSV propio
        """
    )
    
    parser.add_argument('--entrenar', action='store_true',
                       help='Entrenar nuevo modelo con dataset simulado')
    parser.add_argument('--predecir', action='store_true',
                       help='Predecir perfil con modelo existente')
    parser.add_argument('--csv', type=str,
                       help='Ruta al archivo CSV para entrenamiento')
    
    args = parser.parse_args()
    
    # Determinar modo de ejecución
    if args.entrenar:
        mostrar_banner()
        modo_entrenar(args.csv)
    elif args.predecir:
        mostrar_banner()
        modo_predecir()
    elif args.csv:
        mostrar_banner()
        modo_entrenar(args.csv)
    else:
        # Modo interactivo por defecto
        modo_interactivo()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Programa interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
