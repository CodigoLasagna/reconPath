import cv2
import mediapipe as mp
import os
import csv
import time
import threading
from PIL import ImageTk, Image

class HandGestureDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, auto_word='', classifier=None, cap=None, app=None, dataset_path='', show_webcam=False):
        self.mp_hands = mp.solutions.hands
        self.classifier = classifier
        self.cap = cap
        self.app = app
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.dataset_path = "datasets"
        os.makedirs(self.dataset_path, exist_ok=True)
        self.csv_path = dataset_path
        self.photo_counter = 0
        self.pics_to_take_n = 0
        self.auto_word = auto_word
        self.frame = None
        self.results = None
        self.capturing_timer = False
        self.countdown_time_current = 0
        self.show_webcam = show_webcam
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['photo_path', 'gesture_label', 'keypoints_left', 'keypoints_right'])
    def _cover_webcam(self):
        if (self.show_webcam == False):
            height, width, _ = self.frame.shape
            cv2.rectangle(self.frame, (0, 0), (width, height), (0, 0), -1)


    def detect_gestures(self, frame_rgb):
        self.frame = frame_rgb
        self.results = self.hands.process(self.frame)
        self._cover_webcam()


        if self.results.multi_hand_landmarks:
            keypoints_combined = {}
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_type = handedness.classification[0].label  # 'Left' o 'Right'
                keypoints = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                keypoints_combined[hand_type] = keypoints
                
                self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if self.classifier and self.classifier.knn_model:
                features = self.classifier.extract_combined_features(keypoints_combined)
                prediction = self.classifier.knn_model.predict([features])
        
                cv2.putText(self.frame, f'Gesto: {prediction}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 0, 77), 2, cv2.LINE_AA)
                self.app.current_output = prediction

        if (self.capturing_timer == True):
            cv2.putText(self.frame, f"Capturando en {self.countdown_time_current}...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 0, 77), 2, cv2.LINE_AA)


        img = Image.fromarray(self.frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.app.update_video_label(imgtk)

    def _save_snapshot(self):
        if self.auto_word == '':
            return
        else:
            gesture_label = self.auto_word
            
        photo_path = os.path.join(self.dataset_path, f"{gesture_label}_{self.photo_counter}.png")
        #frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(photo_path, frame_rgb)
        print(f"Foto guardada: {photo_path}")
        self._save_label(photo_path, gesture_label, self.results)
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
            self.classifier.load_cvs_data()
            self.classifier.train()
            self.classifier.save_model()
            self.classifier.load_model()
            self.classifier.evaluate()

    def _change_output_word(self):
        self.auto_word = input("Introduzca el nombre para la etiqueta del gesto: ")

    def _timed_capture(self):
        countdown_time = 5
        self.capturing_timer = True

        def countdown():
            for i in range(countdown_time, 0, -1):
                # Copiar el frame original
                #frame_with_text = self.frame.copy()
                self.countdown_time_current = i
                
                #img = Image.fromarray(frame_with_text)
                #imgtk = ImageTk.PhotoImage(image=img)
                # Actualizar la etiqueta de video en la GUI
                #self.app.update_video_label(imgtk)
                
                # Esperar 1 segundo entre actualizaciones
                time.sleep(1)
            
            self.capturing_timer = False
            self._save_snapshot()

        # Iniciar la cuenta regresiva en un hilo separado para no bloquear la interfaz de usuario
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()

    def _timed_capture_maxim(self):
        countdown_time = 5
        self.capturing_timer = True

        def countdown():
            for i in range(countdown_time, 0, -1):
                # Copiar el frame original
                #frame_with_text = self.frame.copy()
                self.countdown_time_current = i
                
                #img = Image.fromarray(frame_with_text)
                #imgtk = ImageTk.PhotoImage(image=img)
                # Actualizar la etiqueta de video en la GUI
                #self.app.update_video_label(imgtk)
                
                # Esperar 1 segundo entre actualizaciones
                time.sleep(1)
            
            self.capturing_timer = False
            self._save_snapshot()
            self._timed_capture_custom(1)

        # Iniciar la cuenta regresiva en un hilo separado para no bloquear la interfaz de usuario
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()

    def _timed_capture_custom(self, countdown_time):
        self.capturing_timer = True

        def countdown():
            for i in range(countdown_time, 0, -1):
                #frame_with_text = self.frame.copy()
                self.countdown_time_current = i
                
                #img = Image.fromarray(frame_with_text)
                #imgtk = ImageTk.PhotoImage(image=img)
                #self.app.update_video_label(imgtk)
                
                time.sleep(0.05)
            
            self.capturing_timer = False
            self._save_snapshot()
            self.pics_to_take_n -= 1
            if (self.pics_to_take_n >=2 ):
                self._timed_capture_custom(1)

        # Iniciar la cuenta regresiva en un hilo separado para no bloquear la interfaz de usuario
        countdown_thread = threading.Thread(target=countdown)
        countdown_thread.start()
