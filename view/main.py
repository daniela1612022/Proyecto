import os
import tkinter as tk
from tkinter import messagebox
import subprocess

# Ruta al intérprete de Python en tu entorno virtual
python_interpreter = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'venv', 'Scripts', 'python.exe'))

def run_hand_detection():
    try:
        subprocess.Popen([python_interpreter, os.path.join(os.path.dirname(__file__), "hand_detection.py")], shell=True)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo ejecutar hand_detection.py: {e}")

def run_hand_drawing():
    try:
        subprocess.Popen([python_interpreter, os.path.join(os.path.dirname(__file__), "hand_drawing.py")], shell=True)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo ejecutar hand_drawing.py: {e}")

def run_number_detection():
    try:
        subprocess.Popen([python_interpreter, os.path.join(os.path.dirname(__file__), "number_detection.py")], shell=True)
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo ejecutar number_detection.py: {e}")

# Crear la ventana principal
root = tk.Tk()
root.title("Proyecto de Machine Learning")

# Crear un marco para la bienvenida
frame_welcome = tk.Frame(root)
frame_welcome.pack(pady=20)

# Etiqueta de bienvenida
label_welcome = tk.Label(frame_welcome, text="Bienvenido a nuestro proyecto de Machine Learning", font=("Helvetica", 16))
label_welcome.pack()

# Crear un marco para los botones
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=20)

# Botón para ejecutar hand_detection.py
button_hand_detection = tk.Button(frame_buttons, text="Detección de Mano", command=run_hand_detection, width=30)
button_hand_detection.pack(pady=5)

# Botón para ejecutar hand_drawing.py
button_hand_drawing = tk.Button(frame_buttons, text="Dibujo con Mano", command=run_hand_drawing, width=30)
button_hand_drawing.pack(pady=5)

# Botón para ejecutar number_detection.py
button_number_detection = tk.Button(frame_buttons, text="Detección de Números", command=run_number_detection, width=30)
button_number_detection.pack(pady=5)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
