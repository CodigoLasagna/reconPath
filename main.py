import customtkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
from modules.gesture_detection import HandGestureDetector
from modules.gestKnn_module import HandGestureClassifierKnn
from modules.input_executer_module import InputExecuter
import warnings
import os
import json
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
tk.set_appearance_mode("dark")

class HandGestureApp(tk.CTkFrame):
    def __init__(self, master=None, cap=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.cap = cap
        self.current_model = "minecraft"
        self.cache_file_name = "minecraft"
        self.allow_execute_output = tk.StringVar(value="off")

        self.initialize_detector()
        self.create_widgets()

        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.check_for_cache()
        self.get_gestures_from_csv()
        self.current_output = ""
        self.input_executer = InputExecuter()
    
    def save_cache(self):
        cache_file = 'cache/' + self.cache_file_name + '.json'
        data = {
            "file_names": {
                "current_model": self.current_model,
            }
        }
        with open(cache_file, 'w') as json_file:
            json.dump(data, json_file)
    
    def check_for_cache(self):
        cache_file = 'cache/' + self.cache_file_name + '.json'
        loaded_data = None
        if not (os.path.exists(cache_file)):
            print("no cache file")
            return
        print("existing file found")
        with open(cache_file, 'r') as json_file:
            loaded_data = json.load(json_file)
        self.current_model = loaded_data['file_names'].get('current_model')

    def initialize_detector(self):
        dataset_path = 'datasets/' + self.current_model + '.csv'
        model_use_path = 'trained_models/' + self.current_model + '.pkl'
        model_train_path = 'trained_models/' + self.current_model + '.pkl'
        self.classifier = initialize_classifier(dataset_path, model_use_path, model_train_path)
        self.detector = HandGestureDetector(classifier=self.classifier, cap=self.cap, app=self, dataset_path=dataset_path)

    def create_widgets(self):
        self.main_frame = tk.CTkFrame(self)
        self.main_frame.pack(expand=True, pady=10, padx=10)

        self.up_frame = tk.CTkFrame(self.main_frame)
        self.up_frame.pack(side="top", pady=5)
        
        self.video_label = tk.CTkLabel(self.up_frame)
        self.video_label.pack(side="left", pady=5, padx=5)
        self.video_label.configure(text='')

        button_frame = tk.CTkFrame(self.up_frame)
        button_frame.pack(side="left", padx=10)

        self.quit_button = tk.CTkButton(button_frame, text="Salir", command=self.master.destroy)
        self.quit_button.pack(pady=5)

        self.take_pic_button = tk.CTkButton(button_frame, text="Tomar foto", command=self.take_pic)
        self.take_pic_button.pack(pady=5)

        self.take_pic_temp_button = tk.CTkButton(button_frame, text="Tomar foto (cronometrado)", command=self.take_timed_pic)
        self.take_pic_temp_button.pack(pady=5)

        self.take_mult_cron_pics = tk.CTkButton(button_frame, text="Tomar fotos (cronometrado) N veces", command=self.take_max_pics_n)
        self.take_mult_cron_pics.pack(pady=5)
        self.input_text_n = tk.CTkTextbox(button_frame, height=5, width=90)
        self.input_text_n.pack()

        self.train_button = tk.CTkButton(button_frame, text="Entrenar modelo", command=self.call_train_model)
        self.train_button.pack(pady=5)

        self.current_word_lbl = tk.CTkLabel(button_frame, text=self.detector.auto_word)
        self.current_word_lbl.pack()
        self.current_word_lbl.configure(text='Etiqueta')

        self.input_text = tk.CTkTextbox(button_frame, height=5, width=90)
        self.input_text.pack()

        self.hide_cam_btn = tk.CTkButton(button_frame, text="Esconder webcam", command=self.hide_cam_toggle)
        self.hide_cam_btn.pack(pady=5)

        images_frame = tk.CTkFrame(self.main_frame)
        images_frame.pack(side="top", pady=5, padx=20)

        self.image_label1 = tk.CTkLabel(images_frame)
        self.image_label1.pack(side="left", padx=10, pady=10)
        self.image_label1.configure(text='')

        self.image_label2 = tk.CTkLabel(images_frame)
        self.image_label2.pack(side="right", padx=10, pady=10)
        self.image_label2.configure(text='')


        #table_inputs_frame = tk.CTkFrame(up_frame)
        #table_inputs_frame.pack(side="right", padx=10)

        self.load_images()
        self.prepare_table()

        self.exe_out_btn = tk.CTkSwitch(button_frame, text="Ejecutar output", onvalue="on",offvalue="off", variable=self.allow_execute_output)
        self.exe_out_btn.pack(pady=5)

    def prepare_table(self):
        #prepare main frame
        self.table_inputs_frame = tk.CTkFrame(self.up_frame)
        self.table_inputs_frame.pack(side="right", padx=10)
        #prepare top frame
        self.table_top_frame = tk.CTkFrame(self.table_inputs_frame)
        self.table_top_frame.pack(side="top", padx=5)


        #create canvas
        self.canvas = tk.CTkCanvas(self.table_top_frame, height=150, width=175, bg = "black")
        self.scrollbar = tk.CTkScrollbar(self.table_top_frame, command=self.canvas.yview)
        self.scrollable_frame = tk.CTkFrame(self.canvas)
        #setup scrollframe
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        #pack canvas
        self.canvas.pack(side="left", expand=True, fill="both")
        self.scrollbar.pack(side="right")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.gesture_inputs_list = []

        self.entries = []
        self.update_table()
        show_button = tk.CTkButton(self.table_inputs_frame, text = "Mostrar valores", command = self.show_values)
        show_button.pack(side="left", padx = 5, pady = 5)
        save_button = tk.CTkButton(self.table_inputs_frame, text = "Guardar valores", command = self.save_inputs_cache_table_safe)
        save_button.pack(side="right", padx = 5, pady = 5)

    def update_table(self):
        self.entries = []
        for i in range(len(self.gesture_inputs_list)):
            row = []
            #default column (for model gestures)
            label = tk.CTkLabel(self.scrollable_frame, text=self.gesture_inputs_list[i])
            label.grid(row = i, column=0, padx=5, pady=5)
            #input column (for input events)
            entry = tk.CTkEntry(self.scrollable_frame)
            entry.grid(row = i, column = 1, padx = 5, pady = 5)
            row.append(entry)

            self.entries.append(row)


    def show_values(self):
        for i, row in enumerate(self.entries):
            print(f"{self.gesture_inputs_list[i]}: {row[0].get()}")
            #row[0].delete(0, tk.END)
            #row[0].insert(0, "hi")


    def hide_cam_toggle(self):
        self.detector.show_webcam = not (self.detector.show_webcam)

    def call_train_model(self):
        self.detector._train_model()
        self.load_images()
        self.get_gestures_from_csv()

    def get_gestures_from_csv(self):
        csv_file = 'datasets/' + self.current_model + '.csv'
        cache_file_gestures = 'cache/' + self.current_model + "_gestures_list.json"
        if not (os.path.exists(csv_file)):
            return
        df = pd.read_csv(csv_file)
        column = df['gesture_label']
        count_series = df['gesture_label'].value_counts()
        print(count_series)
        unique_values = column.unique()
        self.prev_values = []
        first_oppening = False
        self.gesture_inputs_list = unique_values[:]
        if (len(self.entries) == 0):
            first_oppening = True

        if (first_oppening):
            if (os.path.exists(cache_file_gestures)):
                with open(cache_file_gestures, 'r') as file:
                    self.prev_values = json.load(file)
        else:
            for i, row in enumerate(self.entries):
                self.prev_values.append(row[0].get())

        self.update_table()

        if (first_oppening):
            if (len(self.prev_values) > 0):
                self.save_inputs_cache()
        else:
            self.save_inputs_cache()
    
    def save_inputs_cache(self):
        cache_file_gestures = 'cache/' + self.current_model + "_gestures_list.json"
        with open(cache_file_gestures, 'w') as file:
            json.dump(self.prev_values, file)
        for i, row in enumerate(self.entries):
            row[0].delete(0, tk.END)
            if (i < len(self.prev_values)):
                row[0].insert(0, self.prev_values[i])
            else:
                row[0].insert(0, '')

    def save_inputs_cache_table_safe(self):
        self.prev_values = []
        for i, row in enumerate(self.entries):
            self.prev_values.append(row[0].get())
        self.save_inputs_cache()



    def take_pic(self):
        self.detector.auto_word = self.input_text.get(1.0, 'end-1c')
        self.detector._save_snapshot()

    def take_timed_pic(self):
        self.detector.auto_word = self.input_text.get(1.0, 'end-1c')
        self.detector._timed_capture()

    def take_max_pics_n(self):
        self.detector.auto_word = self.input_text.get(1.0, 'end-1c')
        self.detector.pics_to_take_n = int(self.input_text_n.get(1.0, 'end-1c'))
        self.detector._timed_capture_maxim()

    def load_images(self):
        conf_matrix_path = "figures/conf_mat.png"
        clas_rep_path = "figures/clas_rep.png"
        if not os.path.isfile(conf_matrix_path) or not os.path.isfile(clas_rep_path):
            return

        img1 = Image.open(conf_matrix_path)
        img1 = img1.resize((700, 490))
        self.imgtk1 = ImageTk.PhotoImage(img1)
        self.image_label1.configure(image=self.imgtk1)

        img2 = Image.open(clas_rep_path)
        img2 = img2.resize((933, 490))
        self.imgtk2 = ImageTk.PhotoImage(img2)
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
            if (self.allow_execute_output.get() == "on"):
                self.call_input_executor()
    def call_input_executor(self):
        if (self.current_output != ""):
            value = self.prev_values[self.find_value_in_array()]
            #print(value)
            self.input_executer.execute_input(value)
            self.current_output = ""
        else:
            self.input_executer.stop_event()

    def find_value_in_array(self):
        for i in range(len(self.gesture_inputs_list)):
            if self.gesture_inputs_list[i] == self.current_output:
                return i

def initialize_classifier(dataset_path, model_path_to_use, model_path_to_train):
    classifier = HandGestureClassifierKnn(dataset_path, model_path_to_use, model_path_to_train)
    return classifier

if __name__ == "__main__":
    root = tk.CTk()
    root.configure()
    root.title("Recon Path")
    
    cap = cv2.VideoCapture(0)
    app = HandGestureApp(master=root, cap=cap)
    app.pack(expand=True, fill="both")

    root.mainloop()
    app.save_cache()
