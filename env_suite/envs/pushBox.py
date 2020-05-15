import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class pushBox(gym.Env):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def __init__(self, grid_size=10, mode='vector'):
        assert mode in ['vector','image'], f'mode {mode} invalid'
        assert grid_size >= 8
        super(pushBox,self).__init__()
        self.mode = mode
        self.grid_size = grid_size
        self.n_steps = 0
        self.max_n_steps = grid_size * 4

        self.state = np.zeros((2,), dtype=int)
        self.goal = np.zeros((2,), dtype=int)
        self.box = np.zeros((2,), dtype=int)
        self.state_color = np.array([0, 255, 0])
        self.goal_color = np.array([255, 0, 0])
        self.box_color = np.array([0, 0, 255])
        
        self.action_space = spaces.Discrete(4)
        if mode == 'vector':
            high = np.full(4, grid_size, dtype=np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        elif mode == 'image':
            self.observation_space = spaces.Box(0, 255, (grid_size * 4, grid_size * 4, 3), dtype = np.uint8)
    
    def _get_obs(self):
        if self.mode == 'vector':
            return np.concatenate((np.subtract(self.state, self.box), np.subtract(self.box, self.goal)))
        else:
            obs = np.zeros((self.grid_size * 4, self.grid_size * 4, 3), dtype=np.uint8)
            obs[4 * self.state[0] : 4 * self.state[0] + 4,
                            4 * self.state[1] : 4 * self.state[1] + 4,
                            np.where(self.state_color == 255)[0]] = 255
            obs[4 * self.goal[0] : 4 * self.goal[0] + 4,
                            4 * self.goal[1] : 4 * self.goal[1] + 4,
                            np.where(self.goal_color == 255)[0]] = 255
            obs[4 * self.box[0] : 4 * self.box[0] + 4,
                            4 * self.box[1] : 4 * self.box[1] + 4,
                            np.where(self.box_color == 255)[0]] = 255
            return obs

    def reset(self):
        self.n_steps = 0

        self.state = np.random.randint(low=0, high=self.grid_size, size=(2,))
        while True:
            self.goal = np.random.randint(low=1, high=self.grid_size - 1, size=(2,))
            if not np.array_equal(self.goal, self.state):
                break
        while True:
            self.box = np.random.randint(low=1, high=self.grid_size - 1, size=(2,))
            if not (np.array_equal(self.box, self.state) or np.array_equal(self.box, self.goal)):
                break
        return self._get_obs()
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.n_steps += 1

        if action == self.RIGHT:
            self.state = np.add(self.state, np.array([1,0]))
            if np.array_equal(self.state, self.box):
                self.box = np.add(self.box, np.array([1,0]))
        elif action == self.LEFT:
            self.state = np.add(self.state, np.array([-1,0]))
            if np.array_equal(self.state, self.box):
                self.box = np.add(self.box, np.array([-1,0]))
        elif action == self.UP:
            self.state = np.add(self.state, np.array([0,1]))
            if np.array_equal(self.state, self.box):
                self.box = np.add(self.box, np.array([0,1]))
        elif action == self.DOWN:
            self.state = np.add(self.state, np.array([0,-1]))
            if np.array_equal(self.state, self.box):
                self.box = np.add(self.box, np.array([0,-1]))
        
        self.state = np.clip(self.state, 0, self.grid_size - 1)

        done = False
        info = {}
        reward = 0
        reward -= np.sum(np.abs(np.subtract(self.box, self.state))) / (8 * self.grid_size)
        reward -= np.sum(np.abs(np.subtract(self.goal, self.box))) / (4 * self.grid_size)

        if np.array_equal(self.box, self.goal):
            reward = 1
            done = True
        elif 0 in self.box or (self.grid_size - 1) in self.box:
            reward = -1
            done = True
        elif self.n_steps == self.max_n_steps:
            done = True
        
        return self._get_obs(), reward, done, info

    def render(self, mode='console'):
        assert mode == 'console', "Invalid mode for rendering"
        
        print(" ", end="")
        for _ in range(self.grid_size + 2):
            print("■ ", end="")
        print("")
        for y in range(self.grid_size - 1, -1, -1):     # y in [0,9]
            print(" ■ ", end="")
            for x in range(self.grid_size):
                cell = np.array([x,y])
                if np.array_equal(cell, self.state):
                    print("x ", end="")
                elif np.array_equal(cell, self.box):
                    print("□ ", end="")
                elif np.array_equal(cell, self.goal):
                    print("G ", end="")
                else:
                    print(". ", end="")
            print("■")
        print(" ", end="")
        for _ in range(self.grid_size + 2):
            print("■ ", end="")
        print("")

    def close(self):
        pass