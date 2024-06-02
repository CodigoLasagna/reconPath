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
        self.video_label = tk.Label(self, borderwidth=2, relief="solid")
        self.video_label.pack(side="left", pady=10)  # Adjust padx and pady as needed

        # Create a frame for the buttons
        button_frame = tk.Frame(self)
        button_frame.pack(side="right", padx=200)

        self.quit_button = tk.Button(button_frame, text="Salir", command=self.master.destroy)
        self.quit_button.pack(padx=10, pady=5)

        self.take_pic_button = tk.Button(button_frame, text="Tomar foto", command=self.detector._save_snapshot)
        self.take_pic_button.pack(padx=10, pady=5)

        self.take_pic_temp_button = tk.Button(button_frame, text="Tomar foto (cronometrado)", command=self.detector._timed_capture)
        self.take_pic_temp_button.pack(padx=10, pady=5)

        self.train_button = tk.Button(button_frame, text="Entrenar modelo", command=self.detector._train_model)
        self.train_button.pack(padx=10, pady=5)

        self.current_word_lbl = tk.Label(button_frame, text=self.detector.auto_word)
        self.current_word_lbl.pack()

        self.input_text = tk.Text(button_frame, height= 5, width= 20);
        self.input_text.pack()

        self.input_word_btn = tk.Button(button_frame, text="establecer etiqueta", command=self.update_current_train_label)
        self.input_word_btn.pack(padx=10, pady=5)
    def update_current_train_label(self):
        self.detector.auto_word = self.input_text.get(1.0, 'end-1c')
        self.detector.photo_counter = 0
        self.current_word_lbl.config(text=self.detector.auto_word)

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
