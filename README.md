# Recon Path - Aplicación de Detección de Gestos con OpenCV y tkinter

Recon Path es una aplicación desarrollada en Python que utiliza OpenCV y tkinter para la detección de gestos de mano en tiempo real y la interacción con el usuario a través de una interfaz gráfica.

## Funcionalidades

- **Detección de Gestos en Tiempo Real**: Utiliza la biblioteca MediaPipe para detectar gestos de mano en el video de entrada de la cámara.
  
- **Clasificación de Gestos**: Utiliza un clasificador KNN (K-Nearest Neighbors) entrenado para clasificar los gestos detectados.
  
- **Interfaz Gráfica con tkinter**: Proporciona una interfaz de usuario intuitiva para interactuar con la aplicación, tomar fotos de gestos, entrenar el modelo y visualizar métricas de rendimiento del modelo.

## Requisitos

- Python 3.x
- Paquetes requeridos: `opencv-python`, `mediapipe`, `Pillow`, `numpy`, `joblib`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/recon-path.git
   cd recon-path
   ```


2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:
   ```bash
   python main.py
   ```
