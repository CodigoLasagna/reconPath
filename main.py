import tkinter as tk
from PIL import Image, ImageTk
import cv2
from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifierKnn
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

class HandGestureApp(tk.Frame):
    def __init__(self, master=None, cap=None):
        super().__init__(master, bg='#1B232B')
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
        up_frame = tk.Frame(self, bg='#1B232B')
        up_frame.pack(side="top", pady=10)
        
        self.video_label = tk.Label(up_frame, borderwidth=2, relief="solid", bg='#FFFFFF')
        self.video_label.pack(side="left", pady=10, padx=10)

        button_frame = tk.Frame(up_frame, bg='#1B232B')
        button_frame.pack(side="right", padx=20)

        self.quit_button = tk.Button(button_frame, text="Salir", command=self.master.destroy, bg='#4CAF50', fg='#FFFFFF')
        self.quit_button.pack(padx=10, pady=5)

        self.take_pic_button = tk.Button(button_frame, text="Tomar foto", command=self.detector._save_snapshot, bg='#4CAF50', fg='#FFFFFF')
        self.take_pic_button.pack(padx=10, pady=5)

        self.take_pic_temp_button = tk.Button(button_frame, text="Tomar foto (cronometrado)", command=self.detector._timed_capture, bg='#4CAF50', fg='#FFFFFF')
        self.take_pic_temp_button.pack(padx=10, pady=5)

        self.train_button = tk.Button(button_frame, text="Entrenar modelo", command=self.call_train_model, bg='#4CAF50', fg='#FFFFFF')
        self.train_button.pack(padx=10, pady=5)

        self.current_word_lbl = tk.Label(button_frame, text=self.detector.auto_word, bg='#1B232B', fg='#FFFFFF')
        self.current_word_lbl.pack()

        self.input_text = tk.Text(button_frame, height=5, width=20)
        self.input_text.pack()

        self.input_word_btn = tk.Button(button_frame, text="Establecer etiqueta", command=self.update_current_train_label, bg='#4CAF50', fg='#FFFFFF')
        self.input_word_btn.pack(padx=10, pady=5)

        images_frame = tk.Frame(self, bg='#1B232B')
        images_frame.pack(side="top", pady=20)

        self.image_label1 = tk.Label(images_frame, borderwidth=2, relief="solid", bg='#FFFFFF')
        self.image_label1.pack(side="top", padx=10, pady=10)

        self.image_label2 = tk.Label(images_frame, borderwidth=2, relief="solid", bg='#FFFFFF')
        self.image_label2.pack(side="bottom", padx=10, pady=10)

        self.load_images()
    
    def call_train_model(self):
        self.detector._train_model()
        self.load_images()

    def load_images(self):
        conf_matrix_path = "figures/conf_mat.png"
        clas_rep_path = "figures/clas_rep.png"
        if os.path.isfile(conf_matrix_path) == False:
            return
        if os.path.isfile(clas_rep_path) == False:
            return

        img1 = Image.open(conf_matrix_path)
        imgtk1 = ImageTk.PhotoImage(image=img1)

        img2 = Image.open(clas_rep_path)
        imgtk2 = ImageTk.PhotoImage(image=img2)

        self.image_label1.imgtk = imgtk1
        self.image_label1.config(image=imgtk1)
        self.image_label2.imgtk = imgtk2
        self.image_label2.config(image=imgtk2)

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
    root.configure(bg='#1B232B')
    root.title("Hand Gesture Recognition App")
    
    cap = cv2.VideoCapture(0)
    app = HandGestureApp(master=root, cap=cap)
    app.pack()

    app.start_camera_thread()

    root.mainloop()
