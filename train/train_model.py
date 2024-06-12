import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pickle
import os

# Crear carpeta models si no existe
os.makedirs('../models', exist_ok=True)

# Cargar el conjunto de datos MNIST
print("Cargando el conjunto de datos MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"]/255, mnist["target"]
y = y.astype(np.uint8)  # Convertir etiquetas a enteros

# Dividir el conjunto de datos en entrenamiento y prueba
test_size = 10000  # Tamaño del conjunto de prueba
print(f"Dividiendo el conjunto de datos en {len(X)-test_size} para entrenamiento y {test_size} para prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

# Estandarizar los datos
print("Estandarizando los datos...")
scaler = StandardScaler()  # Crear una instancia de StandardScaler
X_train = scaler.fit_transform(X_train)  # Ajustar y transformar los datos de entrenamiento
X_test = scaler.transform(X_test)  # Transformar los datos de prueba usando el mismo escalador
with open('../models/scaler.pkl', 'wb') as f:  # Guardar el escalador en un archivo
    pickle.dump(scaler, f)

# Entrenar el modelo SVM sin PCA
print("Entrenando el modelo SVM sin PCA...")
classifier_no_pca = SVC(kernel="linear")  # Crear una instancia del clasificador SVM con kernel lineal
classifier_no_pca.fit(X_train, y_train)  # Entrenar el modelo con los datos de entrenamiento
with open('../models/svc_digit_classifier_no_pca.pkl', 'wb') as f:  # Guardar el modelo entrenado en un archivo
    pickle.dump(classifier_no_pca, f)

# Predicciones y evaluación sin PCA
print("Evaluando el modelo SVM sin PCA...")
predicted_no_pca = classifier_no_pca.predict(X_test)  # Hacer predicciones en el conjunto de prueba
print(f"Reporte de clasificación para el clasificador sin PCA:\n"
      f"{metrics.classification_report(y_test, predicted_no_pca)}\n")  # Imprimir el reporte de clasificación

# Guardar las predicciones
with open('../models/predicciones_no_pca.pkl', 'wb') as f:
    pickle.dump(predicted_no_pca, f)

print("Proceso completado y modelos guardados.")
