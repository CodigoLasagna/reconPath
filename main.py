import tkinter as tk
from PIL import Image, ImageTk
import cv2
from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifierKnn

class HandGestureApp(tk.Frame):
    def __init__(self, master=None, cap=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.cap = cap

        self.initialize_detector()
        self.create_widgets()

    def initialize_detector(self):
        dataset_path = 'hand_gesture_dataset/labels.csv'
        model_use_path = 'trained_models/model_01.pkl'
        model_train_path = 'trained_models/model_01.pkl'
        self.classifier = initialize_classifier(dataset_path, model_use_path, model_train_path)
        self.detector = HandGestureDetector(classifier=self.classifier, cap=self.cap, app=self)

    def create_widgets(self):
        self.video_label = tk.Label(self)
        self.video_label.pack()

        self.quit_button = tk.Button(self, text="Salir", command=self.master.destroy)
        self.quit_button.pack()

    def update_video_label(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def start_camera_thread(self):
        def camera_loop():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.detector.detect_gestures(frame_rgb)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.update_video_label(imgtk)
            self.after(10, camera_loop)
    
        camera_loop()

def initialize_classifier(dataset_path, model_path_to_use, model_path_to_train):
    classifier = HandGestureClassifierKnn(dataset_path, model_path_to_use, model_path_to_train)
    return classifier

if __name__ == "__main__":
    root = tk.Tk()
    cap = cv2.VideoCapture(0)
    app = HandGestureApp(master=root, cap=cap)
    app.pack()

    app.start_camera_thread()

    root.mainloop()
