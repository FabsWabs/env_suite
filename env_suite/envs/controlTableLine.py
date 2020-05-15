import numpy as np
from scipy.interpolate import interp1d
import gym
from gym import spaces
from gym.utils import seeding

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