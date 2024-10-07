import pyautogui
import keyboard

pyautogui.PAUSE = 0

class InputExecuter:
    def __init__(self):
        self.cur_event = ''
        self.events = []

    def execute_input(self, input=''):
        if input and input not in self.events:
            self.events.append(input)

        if self.cur_event != input:
            self.stop_event()
            self.cur_event = input

        if input and input[0] == "T":
            self.call_key_event()

        print(self.cur_event)

    def call_key_event(self):
        if len(self.cur_event) > 2:
            keyboard.press(self.cur_event[2])

    def stop_event(self):
        print("events_stopped")
        for event in self.events:
            if len(event) > 2 and event[0] == "T":
                keyboard.release(event[2])

        if len(self.cur_event) > 2:
            keyboard.release(self.cur_event[2])

