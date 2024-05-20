import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv

class HandGestureDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.9):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.dataset_path = "hand_gesture_dataset"
        os.makedirs(self.dataset_path, exist_ok=True)
        self.csv_path = os.path.join(self.dataset_path, "labels.csv")
        self.photo_counter = 0

    def detect_gestures(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    landmarks = hand_landmarks.landmark
                    thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    thumb_index_distance = np.linalg.norm(
                        np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])
                    )

                    if thumb_index_distance < 0.05:
                        cv2.putText(frame, 'Gesto: OK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Recognition', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:  # Presiona 'Esc' para salir
                break
            elif key & 0xFF == ord('p'):  # Presiona 'p' para tomar una foto
                gesture_label = input("Introduce la etiqueta para este gesto: ")
                photo_path = os.path.join(self.dataset_path, f"{gesture_label}_{self.photo_counter}.png")
                cv2.imwrite(photo_path, frame)
                print(f"Foto guardada: {photo_path}")
                self._save_label(photo_path, gesture_label)
                self.photo_counter += 1
            elif key & 0xFF == ord('c'):  # Presiona 'c' para iniciar la captura cronometrada
                self._timed_capture(cap)

        cap.release()
        cv2.destroyAllWindows()

    def _save_label(self, photo_path, gesture_label):
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([photo_path, gesture_label])

    def _timed_capture(self, cap):
        gesture_label = input("Introduce la etiqueta para este gesto: ")
        start_time = time.time()
        end_time = start_time + 10  # Capturar después de 10 segundos

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Mostrar la cuenta regresiva
            remaining_time = int(end_time - time.time())
            cv2.putText(frame, f'Tiempo restante: {remaining_time}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
                break

        # Tomar la foto al finalizar la cuenta regresiva
        ret, final_frame = cap.read()
        if ret:
            photo_path = os.path.join(self.dataset_path, f"{gesture_label}_{self.photo_counter}.png")
            cv2.imwrite(photo_path, final_frame)
            print(f"Foto guardada: {photo_path}")
            self._save_label(photo_path, gesture_label)
            self.photo_counter += 1
