import cv2
from joblib import load
import warnings
import pickle

# Ignorar advertencias de características no válidas
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar el modelo de clasificación y el escalador
with open('../models/svc_digit_classifier_no_pca.pkl', 'rb') as f:
    model = pickle.load(f)
scaler = load('../models/scaler.pkl')

# Inicializar la captura de video
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def prediction(image, model, scaler):
    """
    Realiza una predicción del dígito en una imagen dada utilizando el modelo y el escalador proporcionados.
    
    Args:
        image (numpy.ndarray): Imagen en escala de grises del dígito.
        model: Modelo de clasificación entrenado.
        scaler: Escalador para normalizar las características de la imagen.

    Returns:
        int: Predicción del dígito.
    """
    img = cv2.resize(image, (28, 28))  # Redimensionar a 28x28 píxeles
    img = img.flatten().reshape(1, -1)  # Aplanar y redimensionar para el modelo
    img = scaler.transform(img)  # Escalar las características
    predict = model.predict(img)  # Realizar la predicción
    return predict[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()

    # Definir el tamaño y la posición del cuadro de la región de interés (ROI)
    bbox_size = (100, 100)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    # Recortar y convertir a escala de grises la región de interés
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Umbralización binaria inversa

    # Realizar la predicción del dígito
    digit = prediction(thresh, model, scaler)
    
    # Añadir la predicción del dígito en el frame con letras blancas y sombra negra
    text = f'Digit: {digit}'
    org = (bbox[0][0], bbox[0][1] - 10)  # Posición del texto justo arriba de la ROI
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    shadow_color = (0, 0, 0)

    # Añadir sombra al texto
    cv2.putText(frame_copy, text, (org[0] + 2, org[1] + 2), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
    # Añadir texto blanco
    cv2.putText(frame_copy, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Dibujar el cuadro de la región de interés (ROI) con el preprocesamiento en el cuadro
    frame_copy[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (255, 255, 0), 2)  # Dibujar el cuadro de la ROI en azul cian
    
    # Mostrar la imagen procesada
    cv2.imshow("input", frame_copy)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
