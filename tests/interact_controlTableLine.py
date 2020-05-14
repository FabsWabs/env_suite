from env_suite.env_suite import controlTableLine
import time
import numpy as np

global isWindows

isWindows = False
try:
    from win32api import STD_INPUT_HANDLE
    from win32console import GetStdHandle, KEY_EVENT, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT
    isWindows = True
except ImportError as e:
    import sys
    import select
    import termios


class KeyPoller():
    def __enter__(self):
        global isWindows
        if isWindows:
            self.readHandle = GetStdHandle(STD_INPUT_HANDLE)
            self.readHandle.SetConsoleMode(ENABLE_LINE_INPUT|ENABLE_ECHO_INPUT|ENABLE_PROCESSED_INPUT)

            self.curEventLength = 0
            self.curKeysLength = 0

            self.capturedChars = []
        else:
            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        return self

    def __exit__(self, type, value, traceback):
        if isWindows:
            pass
        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def poll(self):
        if isWindows:
            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)

            eventsPeek = self.readHandle.PeekConsoleInput(10000)

            if len(eventsPeek) == 0:
                return None

            if not len(eventsPeek) == self.curEventLength:
                for curEvent in eventsPeek[self.curEventLength:]:
                    if curEvent.EventType == KEY_EVENT:
                        if ord(curEvent.Char) == 0 or not curEvent.KeyDown:
                            pass
                        else:
                            curChar = str(curEvent.Char)
                            self.capturedChars.append(curChar)
                self.curEventLength = len(eventsPeek)

            if not len(self.capturedChars) == 0:
                return self.capturedChars.pop(0)
            else:
                return None
        else:
            dr,dw,de = select.select([sys.stdin], [], [], 0)
            if not dr == []:
                return sys.stdin.read(1)
            return None

if __name__ == "__main__":
    env = controlTableLine()
    env.reset()
    env.render()
    with KeyPoller() as keyPoller:
        reward_sum = 0
        steps = 0
        while True:
            c = keyPoller.poll()
            done = False
            action = np.zeros((2,), dtype=np.float32)
            if not c is None:
                if c == "c":
                    break
                elif c == "d":
                    action = np.array([1, 0], dtype=np.float32)
                elif c == "w":
                    action = np.array([0, 1], dtype=np.float32)
                elif c == "a":
                    action = np.array([-1, 0], dtype=np.float32)
                elif c == "s":
                    action = np.array([0, -1], dtype=np.float32)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            steps += 1
            print(f"Reward: {reward}, steps: {steps}, state: {obs}")
            env.render()
            if done:
                env.reset()
                reward_sum = 0
                steps = 0
                time.sleep(0.5)
                env.render()