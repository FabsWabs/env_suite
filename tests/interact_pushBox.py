from __future__ import print_function

from env_suite.env_suite import pushBox
from pynput.keyboard import Key, Listener
import time

ENV = pushBox

class Interact():
    def __init__(self):
        self.env = ENV(grid_size=8, mode='vector')
        self.env.reset()
        self.env.render()

    def on_press(self, key):
        done = False
        action = None
        if key == Key.esc:
            self.env.close()
            return False
        elif key == Key.right:
            action = 0
        elif key == Key.up:
            action = 1
        elif key == Key.left:
            action = 2
        elif key == Key.down:
            action = 3
        if not action == None:
            _, reward, done, _ = self.env.step(action)
            # print(f"Reward: {reward}")
        self.env.render()
        if done:
            self.env.reset()
            time.sleep(0.5)
            self.env.render()

    def on_release(self, key):
        pass
        
    def listen(self):
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

if __name__ == "__main__":
    interact = Interact()
    interact.listen()
    del interact