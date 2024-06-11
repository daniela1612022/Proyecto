import mediapipe as mp
import cv2

# Función para contar dedos levantados
def count_raised_fingers(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]  # Índices de las puntas de los dedos

    # Verificar el pulgar (dependiendo de si es la mano izquierda o derecha)
    if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x:  # Mano izquierda
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)  # Pulgar levantado
        else:
            fingers.append(0)  # Pulgar no levantado
    else:  # Mano derecha
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)  # Pulgar levantado
        else:
            fingers.append(0)  # Pulgar no levantado

    # Verificar otros dedos
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Dedo levantado
        else:
            fingers.append(0)  # Dedo no levantado

    return fingers.count(1)  # Contar dedos levantados

# Función para detección de manos y contador de dedos levantados
def hand_detection():
    print("Iniciando la detección de manos...")
    mp_hands = mp.solutions.hands  # Inicializar el módulo de manos de MediaPipe
    mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo de MediaPipe

    cap = cv2.VideoCapture(0)  # Capturar video desde la cámara
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    # Configurar el modelo de detección de manos
    with mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()  # Leer un cuadro de la cámara
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
            results = hands.process(image)  # Procesar la imagen para detectar manos

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir de vuelta a BGR para OpenCV
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar los puntos y conexiones de las manos
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # Conexiones
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Puntos azules
                    )
                    fingers_count = count_raised_fingers(hand_landmarks)  # Contar dedos levantados
                    
                    # Determinar la mano (izquierda o derecha) y posicionar el texto adecuadamente
                    if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x:
                        hand_label = 'Mano izquierda'
                        text_position = (int(hand_landmarks.landmark[0].x * image.shape[1]) - 50, int(hand_landmarks.landmark[0].y * image.shape[0]) + 30)
                    else:
                        hand_label = 'Mano derecha'
                        text_position = (int(hand_landmarks.landmark[0].x * image.shape[1]) - 50, int(hand_landmarks.landmark[0].y * image.shape[0]) + 30)

                    # Dibujar la sombra del texto en negro
                    cv2.putText(image, f'{hand_label}: {fingers_count} dedos', (text_position[0] + 1, text_position[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    # Dibujar el texto en blanco
                    cv2.putText(image, f'{hand_label}: {fingers_count} dedos', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Hand Tracking', image)  # Mostrar la imagen en una ventana
            if cv2.waitKey(5) & 0xFF == 27:  # Salir del bucle si se presiona 'ESC'
                break
    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar todas las ventanas
    print("Detección de manos finalizada.")

if __name__ == "__main__":
    hand_detection()  # Ejecutar la función de detección de manos
