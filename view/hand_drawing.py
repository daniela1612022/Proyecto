import cv2
from joblib import load
import warnings
import pickle
import mediapipe as mp
import numpy as np

# Suprimir advertencias específicas de scikit-learn sobre nombres de características no válidos
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar el modelo SVM y el escalador del disco para la predicción de dígitos
with open('../models/svc_digit_classifier_no_pca.pkl', 'rb') as f:
    model = pickle.load(f)
scaler = load('../models/scaler.pkl')

# Inicialización de la biblioteca MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Configurar la captura de video utilizando la cámara predeterminada
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir la región de interés (ROI) para el dibujo de dígitos
roi_top_left = (WIDTH // 2 - 50, HEIGHT // 2 - 50)
roi_bottom_right = (WIDTH // 2 + 50, HEIGHT // 2 + 50)

drawing = False  # Estado inicial para controlar el dibujo
prev_x, prev_y = None, None  # Variables para almacenar la posición previa

# Crear un lienzo en blanco para el dibujo
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
roi_canvas = np.zeros((100, 100), dtype=np.uint8)

def prediction(image, model, scaler):
    """
    Función para predecir el dígito dibujado en una región de la imagen.
    """
    img = cv2.resize(image, (28, 28))  # Redimensionar imagen a 28x28 píxeles
    img = img.flatten().reshape(1, -1)  # Aplanar la imagen para el modelo
    img = scaler.transform(img)  # Normalizar las características de la imagen
    predict = model.predict(img)  # Predecir el dígito
    return predict[0]

# Bucle principal para el procesamiento de video en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejar el frame para que se vea más natural
    frame_copy = frame.copy()

    # Convertir el frame a formato RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar las manos
    result = hands.process(rgb_frame)

    # Si se detectan manos, procesar cada una
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener la posición del dedo índice
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Comprobar si el dedo está dentro de la ROI y dibujar en el canvas
            if roi_top_left[0] <= index_x <= roi_bottom_right[0] and roi_top_left[1] <= index_y <= roi_bottom_right[1]:
                if prev_x is not None and prev_y is not None and drawing:
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 255, 255), 5)
                    cv2.line(roi_canvas, (prev_x - roi_top_left[0], prev_y - roi_top_left[1]), 
                             (index_x - roi_top_left[0], index_y - roi_top_left[1]), 255, 5)
                prev_x, prev_y = index_x, index_y
                cv2.circle(frame, (index_x, index_y), 5, (0, 255, 0), cv2.FILLED)
            else:
                prev_x, prev_y = None, None

    # Dibujar la ROI en el frame y combinar con el canvas
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Mostrar el frame procesado
    cv2.imshow("Paint in the Air", combined)

    # Manejar la entrada del usuario para controlar la aplicación
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir
        break
    elif key == ord('d'):  # Empezar a dibujar
        drawing = True
    elif key == ord('s'):  # Detener el dibujo
        drawing = False
    elif key == ord('f'):  # Realizar predicción
        img_cropped = roi_canvas.copy()
        img_gray = cv2.cvtColor(cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('processed_digit.png', thresh)
        cv2.imshow("Processed ROI", thresh)
        digit = prediction(thresh, model, scaler)
        cv2.putText(combined, f'Digit: {digit}', (roi_top_left[0], roi_top_left[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(f'Predicción: {digit}')
    elif key == ord('c'):  # Limpiar el canvas
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        roi_canvas = np.zeros((100, 100), dtype=np.uint8)

# Liberar recursos y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
hands.close()
