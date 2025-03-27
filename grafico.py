import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
import numpy as np
import time
import random

# Configurar Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buslink-ff307-default-rtdb.firebaseio.com/'
})

NUM_CAMIONES = 50  # Simular 50 camiones

# Función para obtener los datos de Firebase
def get_data():
    ref = db.reference("camiones")
    data = ref.get() or {}

    # Si Firebase devuelve una lista, convertirla en diccionario
    if isinstance(data, list):
        data = {str(i): data[i] for i in range(len(data)) if data[i] is not None}
    
    return data

# Función para generar datos simulados
def generar_datos_simulados(camiones_reales):
    datos = {}

    for i in range(1, NUM_CAMIONES + 1):  # Asegurar 50 camiones
        str_i = str(i)
        if str_i in camiones_reales:  # Si el camión ya está en Firebase, usamos su info
            datos[str_i] = camiones_reales[str_i]
        else:  # Simular camión ficticio
            personas = random.randint(0, 20)
            estado = "Vacío" if personas <= 5 else "Medio" if personas <= 10 else "Lleno"
            datos[str_i] = {
                "personas_detectadas": personas,
                "estado": estado
            }
    
    return datos

# Actualizar gráfico en tiempo real
def update_graph():
    plt.ion()
    
    while True:
        # Obtener datos reales de Firebase
        data_reales = get_data()
        
        # Generar datos simulados incluyendo los reales
        data = generar_datos_simulados(data_reales)

        camiones = list(data.keys())
        personas = [data[c]["personas_detectadas"] for c in camiones]
        estados = [data[c]["estado"] for c in camiones]

        # Colores según el estado
        colores = ['green' if e == "Vacío" else 'yellow' if e == "Medio" else 'red' for e in estados]

        # Dibujar gráfico mejorado
        plt.figure("Ocupación de los camiones", figsize=(12, 6))
        plt.clf()
        plt.bar(camiones, personas, color=colores)
        plt.xlabel("Camión")
        plt.ylabel("Personas detectadas")
        plt.title("Ocupación de los camiones (Datos reales + Simulación)")
        plt.xticks(rotation=90)  # Rota etiquetas para mejor visibilidad
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Líneas guías en el gráfico
        plt.ylim(0, 25)  # Fijar rango de ocupación

        plt.pause(10)  # Refrescar cada 2 segundos
    
    plt.ioff()

if __name__ == "__main__":
    update_graph()
