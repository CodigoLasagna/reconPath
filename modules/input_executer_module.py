import pyautogui
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
            if (self.cur_event[1] == "C"):
                self.call_click_mouse_event()
            if (self.cur_event[1] == "M"):
                self.call_move_mouse_event()

    def call_click_mouse_event(self):
        self.mouse.press(getattr(Button, self.cur_event[3:]))
    def call_move_mouse_event(self):
        if (self.cur_event[3:] == 'left'):
            self.mouse.move(-self.mouse_sens, 0)
        if (self.cur_event[3:] == 'right'):
            self.mouse.move(self.mouse_sens, 0)
        if (self.cur_event[3:] == 'up'):
            self.mouse.move(0, -self.mouse_sens / 2)
        if (self.cur_event[3:] == 'down'):
            self.mouse.move(0, self.mouse_sens / 2)

    def stop_event(self):
        for event in self.events:
            if len(event) > 2 and event[0] == "T":
                pyautogui.keyUp(event[2])
            if len(event) > 3 and event[0] == "M":
                if (event[1] == "C"):
                    self.mouse.release(getattr(Button, event[3:]))

