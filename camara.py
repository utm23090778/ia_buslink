import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import threading
import datetime
import geocoder
import time

# Configurar Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://buslink-ff307-default-rtdb.firebaseio.com/'
})

# Cargar YOLO
try:
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except Exception as e:
    print("Error al cargar YOLO:", e)
    exit()

# Cargar COCO
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Error: Archivo 'coco.names' no encontrado.")
    exit()

# Abrir cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Variables globales
frame_lock = threading.Lock()
latest_frame = None
numero_camion = 1  # La cámara representa al camión 1
max_capacidad = 15
ultima_ubicacion = None
ultimo_tiempo_ubicacion = time.time()

# Obtener ubicación cada 30 segundos
def get_location():
    global ultima_ubicacion, ultimo_tiempo_ubicacion
    tiempo_actual = time.time()
    
    if ultima_ubicacion is None or (tiempo_actual - ultimo_tiempo_ubicacion) > 30:
        try:
            g = geocoder.ip('me')
            if g.latlng:
                ultima_ubicacion = g.latlng
                ultimo_tiempo_ubicacion = tiempo_actual
                print(f"📍 Nueva ubicación: {ultima_ubicacion}")
            else:
                print("⚠ No se pudo obtener la ubicación.")
        except Exception as e:
            print(f"⚠ Error al obtener la ubicación: {e}")
    
    return ultima_ubicacion if ultima_ubicacion else [None, None]

# Guardar en Firebase
def save_to_firebase(camion, count):
    now = datetime.datetime.now()
    fecha_hora = now.strftime("%Y-%m-%d %H:%M:%S")
    lat, lon = get_location()
    
    if lat is None or lon is None:
        print("⚠ No se pudo obtener la ubicación. No se guardará.")
        return

    data = {
        "personas_detectadas": count,
        "fecha_hora": fecha_hora,
        "ubicacion": {"latitud": lat, "longitud": lon},
        "estado": "Vacío" if count <= 5 else "Medio" if count <= 10 else "Lleno",
        "asientos_disponibles": max(0, max_capacidad - count)
    }

    db.reference(f"camiones/{camion}").set(data)
    print(f"✅ Camión {camion}: {data}")

# Procesar frames
def process_frame():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.2)
                continue
            frame = latest_frame.copy()

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        height, width, _ = frame.shape
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        person_count = sum(1 for i in indices.flatten() if classes[class_ids[i]] == "person")
        save_to_firebase(numero_camion, person_count)
        
        # Ampliar ventana de la cámara
        cv2.namedWindow("Detección de Personas - Camión 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detección de Personas - Camión 1", 900, 600)  # Tamaño más grande
        cv2.putText(frame, f"Camión {numero_camion} - Personas: {person_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Detección de Personas - Camión 1", frame)

        if cv2.waitKey(1) & 0xFF == ord('z'):
            break

# Iniciar hilo para procesar la detección
threading.Thread(target=process_frame, daemon=True).start()

# Capturar frames en el hilo principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame.")
        break
    with frame_lock:
        latest_frame = frame.copy()
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

cap.release()
cv2.destroyAllWindows()
