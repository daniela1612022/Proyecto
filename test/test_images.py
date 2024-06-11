import cv2
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar el modelo SVM sin PCA
with open('models/svc_digit_classifier_no_pca.pkl', 'rb') as f:
    svm_no_pca = pickle.load(f)

# Cargar el modelo SVM con PCA
with open('models/svc_digit_classifier_with_pca.pkl', 'rb') as f:
    svm_with_pca = pickle.load(f)

# Cargar el escalador
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar los componentes PCA
with open('models/pca_components.pkl', 'rb') as f:
    pca_components = pickle.load(f)

def preprocess_digit(roi):
    """Preprocesar la región de interés (ROI) para que sea compatible con el modelo."""
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype('float32') / 255.0
    roi = roi.flatten().reshape(1, -1)
    roi = scaler.transform(roi)
    return roi

def preprocess_digit_pca(roi):
    """Preprocesar la región de interés (ROI) para que sea compatible con el modelo con PCA."""
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype('float32') / 255.0
    roi = roi.flatten().reshape(1, -1)
    roi = scaler.transform(roi)
    roi = np.dot(roi - np.mean(roi), pca_components)
    return roi

def predict_digit(image_path):
    """Predecir el dígito en una imagen dada utilizando ambos modelos SVM."""
    # Leer la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen {image_path}. Verifique la ruta del archivo.")
    
    # Invertir los colores para que los trazos negros se vean blancos y el fondo negro
    _, thresh = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
    
    # Preprocesar la imagen
    roi_preprocessed = preprocess_digit(thresh)
    prediction_no_pca = svm_no_pca.predict(roi_preprocessed)[0]

    roi_preprocessed_pca = preprocess_digit_pca(thresh)
    prediction_with_pca = svm_with_pca.predict(roi_preprocessed_pca)[0]

    return prediction_no_pca, prediction_with_pca

# Lista de imágenes para probar
image_paths = [
    'test/0.jpg',
    'test/2.jpg',
    'test/2n.jpg',
    'test/3.jpg',
    'test/4.jpg',
    'test/5.jpg',
    'test/9.jpg'
]

# Realizar predicciones en las imágenes
predictions = {}
for image_path in image_paths:
    try:
        predictions[image_path] = predict_digit(image_path)
    except ValueError as e:
        print(e)

print(predictions)
