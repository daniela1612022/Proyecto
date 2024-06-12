import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
with open('../models/scaler_pca.pkl', 'wb') as f:  # Guardar el escalador en un archivo
    pickle.dump(scaler, f)

# Aplicar PCA para reducir la dimensionalidad
print("Aplicando PCA para reducir la dimensionalidad...")
pca = PCA(n_components=0.95, svd_solver='full')  # Crear una instancia de PCA que conserve el 95% de la varianza
X_train_pca = pca.fit_transform(X_train)  # Ajustar PCA y transformar los datos de entrenamiento
X_test_pca = pca.transform(X_test)  # Transformar los datos de prueba usando el mismo PCA
with open('../models/pca.pkl', 'wb') as f:  # Guardar el modelo PCA en un archivo
    pickle.dump(pca, f)

# Entrenar el modelo SVM con PCA
print("Entrenando el modelo SVM con PCA...")
classifier = SVC(kernel="linear")  # Crear una instancia del clasificador SVM con kernel lineal
classifier.fit(X_train_pca, y_train)  # Entrenar el modelo con los datos de entrenamiento reducidos
with open('../models/svc_digit_classifier_with_pca.pkl', 'wb') as f:  # Guardar el modelo entrenado en un archivo
    pickle.dump(classifier, f)

# Predicciones y evaluación con PCA
print("Evaluando el modelo SVM con PCA...")
predicted = classifier.predict(X_test_pca)  # Hacer predicciones en el conjunto de prueba reducido
print(f"Reporte de clasificación para el clasificador con PCA:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")  # Imprimir el reporte de clasificación

# Guardar las predicciones
print("Guardando las predicciones del modelo SVM con PCA...")
with open('../models/predicciones_with_pca.pkl', 'wb') as f:
    pickle.dump(predicted, f)

print("Proceso completado y modelos guardados.")
