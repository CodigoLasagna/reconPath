import cv2
import mediapipe as mp
import os
import csv
import time
from PIL import ImageTk, Image

class HandGestureDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.9, auto_word='', classifier=None, cap=None, app=None):
        self.mp_hands = mp.solutions.hands
        self.classifier = classifier
        self.cap = cap  # Guardar la instancia de captura de cámara
        self.app = app  # Referencia a la instancia de la app tkinter
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.dataset_path = "hand_gesture_dataset"
        os.makedirs(self.dataset_path, exist_ok=True)
        self.csv_path = os.path.join(self.dataset_path, "labels.csv")
        self.photo_counter = 0
        self.auto_word = auto_word
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['photo_path', 'gesture_label', 'keypoints_left', 'keypoints_right'])

    def detect_gestures(self, frame_rgb):
        # Procesar el fotograma RGB
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            keypoints_combined = {}
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label  # 'Left' o 'Right'
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                keypoints_combined[hand_type] = keypoints
                
                # Dibujar landmarks utilizando Mediapipe
                self.mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Clasificar gesto utilizando el modelo cargado
            if self.classifier and self.classifier.knn_model:
                features = self.classifier.extract_combined_features(keypoints_combined)
                prediction = self.classifier.knn_model.predict([features])
        
                # Dibujar texto del gesto
                cv2.putText(frame_rgb, f'Gesto: {prediction}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 0, 77), 2, cv2.LINE_AA)


        # No usar cv2.imshow aquí para evitar conflictos con tkinter
        # Enviar el nuevo fotograma a la app tkinter para que lo actualice
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.app.update_video_label(imgtk)

    def _save_snapshot(self, frame, gesture_label, results):
        if self.auto_word == '':
            gesture_label = input("Introduce la etiqueta para este gesto: ")
        else:
            gesture_label = self.auto_word
            
        photo_path = os.path.join(self.dataset_path, f"{gesture_label}_{self.photo_counter}.png")
        cv2.imwrite(photo_path, frame)
        print(f"Foto guardada: {photo_path}")
        self._save_label(photo_path, gesture_label, results)
        self.photo_counter += 1

    def _save_label(self, photo_path, gesture_label, results):
        keypoints_combined = {'Left': [], 'Right': []}
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label  # 'Left' o 'Right'
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                keypoints_combined[hand_type] = keypoints
            
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([photo_path, gesture_label, keypoints_combined['Left'], keypoints_combined['Right']])

    def _train_model(self):
        if self.classifier:
            self.classifier.train()
            self.classifier.evaluate()
            self.classifier.save_model()

    def _change_output_word(self):
        self.auto_word = input("Introduzca el nombre para la etiqueta del gesto: ")

    def _timed_capture(self, cap):
        if self.auto_word == '':
            gesture_label = input("Introduce la etiqueta para este gesto: ")
        else:
            gesture_label = self.auto_word
        start_time = time.time()
        end_time = start_time + 5

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            remaining_time = int(end_time - time.time())
            cv2.putText(frame, f'Tiempo restante: {remaining_time}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar el fotograma actualizado en la ventana de OpenCV
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Al finalizar el tiempo, guardar la última instantánea capturada
        ret, final_frame = cap.read()
        if ret:
            self._save_snapshot(final_frame, gesture_label, results)

        # Liberar recursos de la cámara y cerrar ventanas de OpenCV
        self.cap.release()
        cv2.destroyAllWindows()
