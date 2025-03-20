import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import threading
import datetime
import geocoder
import time

# 🔹 Configurar Firebase
cred = credentials.Certificate("firebase-key.json")  #llave para firebase
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buslink-ff307-default-rtdb.firebaseio.com/'  # 🔥 Reemplaza con la URL de tu base de datos
})

# 🔹 Cargar los archivos de YOLO con manejo de errores
try:
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print("Error al cargar los archivos de YOLO:", e)
    exit()

# 🔹 Cargar las clases de COCO
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: Archivo 'coco.names' no encontrado.")
    exit()

# 🔹 Abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# 🔹 Variables globales
frame_lock = threading.Lock()
latest_frame = None
person_count = 0
last_saved_count = -1  # Para evitar guardar datos repetitivos

def get_location():
    """Obtiene la ubicación (latitud y longitud) del usuario"""
    g = geocoder.ip('me')
    return g.latlng if g.latlng else [None, None]

def save_to_firebase(count):
    """Guarda los datos en Firebase solo si hay un cambio en la detección"""
    global last_saved_count
    if count == last_saved_count:
        return  # No guardar datos repetidos

    now = datetime.datetime.now()
    fecha_hora = now.strftime("%Y-%m-%d %H:%M:%S")
    lat, lon = get_location()

    if lat is None or lon is None:
        print("⚠️ No se pudo obtener la ubicación.")
        return

    data = {
        "personas_detectadas": count,
        "fecha_hora": fecha_hora,
        "ubicacion": {
            "latitud": lat,
            "longitud": lon
        }
    }
    db.reference("detecciones").push(data)
    last_saved_count = count
    print("✅ Datos guardados en Firebase:", data)

def process_frame():
    """Procesa los frames en un hilo separado"""
    global latest_frame, person_count
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.1)  # Reducir carga del CPU
                continue
            frame = latest_frame.copy()

        # Redimensionar el frame y crear un blob para YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Procesar las detecciones
        height, width, _ = frame.shape
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Filtrar detecciones de baja confianza
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression para eliminar redundancias
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        person_count = 0  # Reiniciar contador

        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                if classes[class_ids[i]] == "person":
                    person_count += 1
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde para personas

        # Guardar en Firebase si hay un cambio en el conteo
        save_to_firebase(person_count)

        # Mostrar el conteo en el frame
        cv2.putText(frame, f"Personas: {person_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen con detecciones
        cv2.imshow("Detección de Personas con YOLO", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Iniciar hilo para procesar frames
thread = threading.Thread(target=process_frame, daemon=True)
thread.start()

# Captura de frames en el hilo principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    with frame_lock:
        latest_frame = frame.copy()

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
