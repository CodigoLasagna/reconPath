import customtkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
from gesture_detection import HandGestureDetector
from gestKnn_module import HandGestureClassifierKnn
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
tk.set_appearance_mode("dark")

class HandGestureApp(tk.CTkFrame):
    def __init__(self, master=None, cap=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.cap = cap

        self.initialize_detector()
        self.create_widgets()

        # Start the camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def initialize_detector(self):
        dataset_path = 'hand_gesture_dataset/labels.csv'
        model_use_path = 'trained_models/model_01.pkl'
        model_train_path = 'trained_models/model_01.pkl'
        self.classifier = initialize_classifier(dataset_path, model_use_path, model_train_path)
        self.detector = HandGestureDetector(classifier=self.classifier, cap=self.cap, app=self)

    def create_widgets(self):
        up_frame = tk.CTkFrame(self)
        up_frame.pack(side="top", pady=10)
        
        self.video_label = tk.CTkLabel(up_frame)
        self.video_label.pack(side="left", pady=10, padx=10)
        self.video_label.configure(text='')

        button_frame = tk.CTkFrame(up_frame)
        button_frame.pack(side="right", padx=20)

        self.quit_button = tk.CTkButton(button_frame, text="Salir", command=self.master.destroy)
        self.quit_button.pack(padx=10, pady=5)

        self.take_pic_button = tk.CTkButton(button_frame, text="Tomar foto", command=self.detector._save_snapshot)
        self.take_pic_button.pack(padx=10, pady=5)

        self.take_pic_temp_button = tk.CTkButton(button_frame, text="Tomar foto (cronometrado)", command=self.detector._timed_capture)
        self.take_pic_temp_button.pack(padx=10, pady=5)

        self.train_button = tk.CTkButton(button_frame, text="Entrenar modelo", command=self.call_train_model)
        self.train_button.pack(padx=10, pady=5)

        self.current_word_lbl = tk.CTkLabel(button_frame, text=self.detector.auto_word)
        self.current_word_lbl.pack()

        self.input_text = tk.CTkTextbox(button_frame, height=5, width=90)
        self.input_text.pack()

        self.input_word_btn = tk.CTkButton(button_frame, text="Establecer etiqueta", command=self.update_current_train_label)
        self.input_word_btn.pack(padx=10, pady=5)

        images_frame = tk.CTkFrame(self)
        images_frame.pack(side="top", pady=20)

        self.image_label1 = tk.CTkLabel(images_frame)
        self.image_label1.pack(side="left", padx=10, pady=10)
        self.image_label1.configure(text='')

        self.image_label2 = tk.CTkLabel(images_frame)
        self.image_label2.pack(side="right", padx=10, pady=10)
        self.image_label2.configure(text='')

        self.load_images()

    def call_train_model(self):
        self.detector._train_model()
        self.load_images()

    def load_images(self):
        conf_matrix_path = "figures/conf_mat.png"
        clas_rep_path = "figures/clas_rep.png"
        if not os.path.isfile(conf_matrix_path) or not os.path.isfile(clas_rep_path):
            return

        img1 = Image.open(conf_matrix_path)
        img1 = img1.resize((600, 420))
        self.imgtk1 = ImageTk.PhotoImage(image=img1)
        self.image_label1.configure(image=self.imgtk1)

        img2 = Image.open(clas_rep_path)
        img2 = img2.resize((800, 420))
        self.imgtk2 = ImageTk.PhotoImage(image=img2)
        self.image_label2.configure(image=self.imgtk2)

    def update_current_train_label(self):
        self.detector.auto_word = self.input_text.get(1.0, 'end-1c')
        self.detector.photo_counter = 0
        self.current_word_lbl.configure(text=self.detector.auto_word)

    def update_video_label(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def camera_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.detector.detect_gestures(frame_rgb)

                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.after(10, self.update_video_label, imgtk)

def initialize_classifier(dataset_path, model_path_to_use, model_path_to_train):
    classifier = HandGestureClassifierKnn(dataset_path, model_path_to_use, model_path_to_train)
    return classifier

if __name__ == "__main__":
    root = tk.CTk()
    root.configure()
    root.title("Hand Gesture Recognition App")
    
    cap = cv2.VideoCapture(0)
    app = HandGestureApp(master=root, cap=cap)
    app.pack()

    root.mainloop()
