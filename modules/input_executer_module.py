import pyautogui
import threading
from pynput.mouse import Button, Controller

pyautogui.PAUSE = 0

class InputExecuter:
    def __init__(self):
        self.cur_event = ''
        self.events = []
        self.screenRes = (3840, 2160)
        self.x, self.y = (0, 0)
        self.mouse = Controller()
        self.current_position = None
        self.mouse_sens = 100
        self.move_thread = None  # Para controlar el hilo actual de movimiento

    def execute_input(self, input=''):
        if input and input not in self.events:
            self.events.append(input)

        if self.cur_event != input:
            self.stop_event()
            self.cur_event = input

        if input and input[0] == "T":
            self.call_key_event()
        if input and input[0] == "M":
            self.call_mouse_event()

    def call_key_event(self):
        if len(self.cur_event) > 2:
            pyautogui.keyDown(self.cur_event[2])

    def call_mouse_event(self):
        if len(self.cur_event) > 3:
            if self.cur_event[1] == "C":
                self.call_click_mouse_event()
            if self.cur_event[1] == "M":
                self.call_move_mouse_event()

    def call_click_mouse_event(self):
        self.mouse.press(getattr(Button, self.cur_event[3:]))

    def smooth_mouse_move(self, dx, dy, steps):
        step_dx = dx / steps
        step_dy = dy / steps
        for i in range(steps):
            self.mouse.move(step_dx, step_dy)
            # No sleep aquí porque se ejecuta en un hilo aparte

    def call_move_mouse_event(self):
        # Detener el hilo de movimiento previo si está en ejecución
        if self.move_thread and self.move_thread.is_alive():
            return  # Evitar iniciar otro hilo si ya hay uno activo

        # Determinar las direcciones
        if self.cur_event[3:] == 'left':
            dx, dy = -self.mouse_sens, 0
        elif self.cur_event[3:] == 'right':
            dx, dy = self.mouse_sens, 0
        elif self.cur_event[3:] == 'up':
            dx, dy = 0, -self.mouse_sens
        elif self.cur_event[3:] == 'down':
            dx, dy = 0, self.mouse_sens
        else:
            return  # Salir si no hay movimiento válido

        # Lanzar el movimiento suave en un hilo separado
        steps = 60  # Número de pasos para suavizar el movimiento
        self.move_thread = threading.Thread(target=self.smooth_mouse_move, args=(dx, dy, steps))
        self.move_thread.start()

    def stop_event(self):
        for event in self.events:
            if len(event) > 2 and event[0] == "T":
                pyautogui.keyUp(event[2])
            if len(event) > 3 and event[0] == "M":
                if event[1] == "C":
                    self.mouse.release(getattr(Button, event[3:]))
        # Aquí también podrías detener el hilo de movimiento si fuera necesario
