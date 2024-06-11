import cv2
import mediapipe as mp
import time
from joblib import load
import warnings
import pickle
import numpy as np

# Ignorar advertencias de características no válidas
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar el modelo de clasificación y el escalador
with open('../models/svc_digit_classifier_no_pca.pkl', 'rb') as f:
    model = pickle.load(f)
scaler = load('../models/scaler.pkl')

# Función de predicción
def prediction(image, model, scaler):
    img = cv2.resize(image, (28, 28))  # Redimensionar a 28x28 píxeles
    img = img.flatten().reshape(1, -1)  # Aplanar y redimensionar para el modelo
    img = scaler.transform(img)  # Escalar las características
    predict = model.predict(img)  # Realizar la predicción
    return predict[0]

# Inicializar mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar variables
drawing = False
drawing_color = (255, 255, 255)  # Color blanco para el dibujo
thickness = 10  # Grosor del trazo

# Configurar la captura de video
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el cuadro de interés (ROI)
roi_top_left = (WIDTH // 2 - 100, HEIGHT // 2 - 100)
roi_bottom_right = (WIDTH // 2 + 100, HEIGHT // 2 + 100)

# Crear una imagen en blanco para dibujar
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Variables para control de rendimiento
prev_frame_time = 0
new_frame_time = 0

# Variables para suavizar el trazo
prev_x, prev_y = None, None

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Procesar la imagen de la cámara con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Dibujar las conexiones de los puntos de referencia de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener coordenadas del dedo índice
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)

                # Verificar si el dedo índice está en el cuadro de interés
                if roi_top_left[0] < x < roi_bottom_right[0] and roi_top_left[1] < y < roi_bottom_right[1]:
                    drawing = True
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, thickness)
                    prev_x, prev_y = x, y
                else:
                    drawing = False
                    prev_x, prev_y = None, None
                
                # Verificar si todos los dedos están cerrados para borrar
                all_fingers_closed = all(
                    hand_landmarks.landmark[i].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                    for i in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                              mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                              mp_hands.HandLandmark.PINKY_TIP]
                )
                if all_fingers_closed:
                    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None

        # Obtener la región de interés (ROI) del canvas
        roi = canvas[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Umbralización binaria inversa

        # Realizar la predicción del dígito
        if np.any(thresh):
            digit = prediction(thresh, model, scaler)
            text = f'Digit: {digit}'
        else:
            text = 'Digit: N/A'

        # Añadir la predicción del dígito en el frame con letras blancas y sombra negra
        org = (roi_top_left[0], roi_top_left[1] - 10)  # Posición del texto justo arriba de la ROI
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        shadow_color = (0, 0, 0)

        # Añadir sombra al texto
        cv2.putText(frame, text, (org[0] + 2, org[1] + 2), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
        # Añadir texto blanco
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Dibujar el cuadro de la región de interés (ROI) con el preprocesamiento en el cuadro
        frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 255, 0), 2)  # Dibujar el cuadro de la ROI en azul cian
        
        # Combinar el canvas con el marco
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Calcular el tiempo por frame para medir FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        # Mostrar el FPS en el frame
        cv2.putText(combined, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar la imagen procesada
        cv2.imshow('Dibujar en el aire', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
