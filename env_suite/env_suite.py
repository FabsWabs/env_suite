import numpy as np
from scipy.interpolate import interp1d
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

class controlTableLine(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, dt=0.1, goal_vel=0.1, min_goal_dist=0.05, max_speed=0.2):
        super(controlTableLine, self).__init__()
        self.max_n_steps = 1000
        self.dt = dt
        self.sup_vals = 5
        self.total_points = 25


        self.max_action = 1.0
        self.max_position = 1.0
        self.max_speed = max_speed
        self.min_goal_dist = min_goal_dist
        self.goal_vel = goal_vel
        self.power = 0.015

        self.low_state = np.array([-self.max_position, -self.max_position, -self.max_speed,
                                    -self.max_speed, -self.max_position * 2, -self.max_position * 2], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_position, self.max_speed,
                                    self.max_speed, self.max_position * 2, self.max_position * 2], dtype=np.float32)
        self.low_action = np.array([-self.max_action, -self.max_action], dtype=np.float32)
        self.high_action = np.array([self.max_action, self.max_action], dtype=np.float32)

        self.action_space = spaces.Box(low=self.low_action,
                                        high=self.high_action, dtype=np.float32)
        self.observation_space = spaces.Box(low = self.low_state,
                                            high = self.high_state, dtype=np.float32)

        self.viewer = None
        
        self.seed()
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.traj_point = 0
        self.trajectory = self.createTrajectory()
        self.state = np.concatenate([self.np_random.uniform(low=[-self.max_position]*2,
                                                            high=[self.max_position]*2), [0,0]])
        return self._get_obs()
    
    def step(self, action):
        self.n_steps += 1
        position = self.state[:2]
        velocity = self.state[2:]
        force = np.clip(action, self.low_action, self.high_action)
        velocity += force * self.power
        position += velocity * self.dt
        abs_distance = np.linalg.norm(self.trajectory[self.traj_point] - position)
        abs_velocity = np.linalg.norm(velocity)

        done = False
        if self.n_steps >= self.max_n_steps:
            reward = -self.max_n_steps
            done = True
        elif True in np.greater(np.abs(position), self.high_state[:2]):
            reward = -self.max_n_steps
            done = True
        elif abs_velocity > self.max_speed:
            reward = -self.max_n_steps
            done = True
        elif abs_distance < self.min_goal_dist:
            self.traj_point += 1
            reward = 1
            if self.traj_point == self.total_points:
                done = True
                self.traj_point -= 1
        else:
            dist_reward = (1 - abs_distance/2) ** 4
            vel_discount = np.abs(abs_velocity - self.goal_vel)
            reward = dist_reward - vel_discount * 20 - 1
            
        position = np.clip(position, self.low_state[:2], self.high_state[:2])
        velocity = np.clip(velocity, self.low_state[2:4], self.high_state[2:4])

        self.state = np.concatenate([position, velocity])
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.state, self.trajectory[self.traj_point] - self.state[:2]])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = 2 * self.max_position
        scale = screen_width/world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            player_circle = rendering.make_circle(10)
            player_circle.set_color(.0, .0, 1.0)
            player_circle.add_attr(
                rendering.Transform(translation=(0, 0))
            )
            self.player = rendering.Transform()
            player_circle.add_attr(self.player)
            self.viewer.add_geom(player_circle)

            traj_circles = []
            self.traj_holder = []
            for i in range(self.total_points):
                traj_circles.append(rendering.make_circle(7))
                if i == 0:
                    traj_circles[i].set_color(1.0, .0,.0)
                else:
                    traj_circles[i].set_color(.0, 1.0,.0)
                traj_circles[i].add_attr(
                    rendering.Transform(translation= (0,0))
                )
                self.traj_holder.append(rendering.Transform())
                traj_circles[i].add_attr(self.traj_holder[i])
                self.viewer.add_geom(traj_circles[i])
                

        pos = self.state[:2]

        self.player.set_translation((pos[0] + self.max_position) * scale, (pos[1] + self.max_position) * scale)
        for i in range(self.total_points):
            if i == 0:
                self.traj_holder[i].set_translation((self.trajectory[self.traj_point,0] + self.max_position) * scale,
                                                    (self.trajectory[self.traj_point,1] + self.max_position) * scale)
            elif i <= self.traj_point:
                self.traj_holder[i].set_translation((self.trajectory[i-1,0] + self.max_position) * scale,
                                                    (self.trajectory[i-1,1] + self.max_position) * scale)
            else:
                self.traj_holder[i].set_translation((self.trajectory[i,0] + self.max_position) * scale,
                                                    (self.trajectory[i,1] + self.max_position) * scale)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def createTrajectory(self):
        points = np.random.uniform(-0.8 ,0.8,(self.sup_vals,2))

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)/distance[-1]

        alpha = np.linspace(0, 1, self.total_points)

        interpolator = interp1d(distance, points, kind='quadratic', axis=0)
        interpolated_points = interpolator(alpha)
        return np.clip(interpolated_points, self.low_state[:2] * 0.9, self.high_state[:2] * 0.9)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None