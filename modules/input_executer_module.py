import pyautogui
pyautogui.PAUSE = 0

class InputExecuter:
    def __init__(self):
        self.cur_event = ''

    def execute_input(self, input = ''):
        self.cur_event = input
        if (input[0] == "T"):
            self.call_key_event()

    def call_key_event(self):
        pyautogui.write(self.cur_event[2])


