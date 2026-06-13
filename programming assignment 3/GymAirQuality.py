import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SensorTransmissionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        #the constant params given
        self.lmda = 0.1
        self.B = 10
        self.eta = 2
        self.Delta = 3

        #loading the probs matrices
        self.air_matrix = np.load('air.npy')
        self.solar_matrix = np.load('solar.npy')

        self.observation_space = spaces.MultiDiscrete([51, 11, 51, 51])             #the sizes are the size of the three theta's and b (B = 10, so it'll be 11)
        self.action_space = spaces.Discrete(3)                                      #only three possible actions (0, 1, 2)

        #initially, just init them to None, cuz actual init takes place when env.reset() is called
        self.theta_t = None
        self.b_t = None
        self.theta_hat = None
        self.theta_max_t = None

        #init is just a placeholder..., the actual reset will be done in .reset() method
        self.time_step = 0       
        
    def step(self, action):
        #checking if transmission is attempted or not, and sampling the success (lamda)
        transmitted = action in [1, 2]
        success = transmitted and (np.random.random() < self.lmda)

        #calc theta_hat, and reward
        if success and action == 1:
            theta_hat_new = self.theta_t
        elif success and action == 2:
            theta_hat_new = self.theta_max_t
        else:
            theta_hat_new = self.theta_hat

        if self.theta_t <= theta_hat_new:
            reward = -(abs(self.theta_t - theta_hat_new))
        else:
            reward = -1.5 * (abs(self.theta_t - theta_hat_new))

        #sampling the next theta_t
        theta_t_idx = round(self.theta_t / 0.02)
        next_theta_t_idx = np.random.choice(51, p=self.air_matrix[theta_t_idx])
        next_theta_t = next_theta_t_idx * 0.02

        # updateing theta_max
        if success:
            theta_max_new = next_theta_t
        else:
            theta_max_new = max(self.theta_max_t, next_theta_t)

        #sample solar energy and hence update battery level
        delta_t = np.random.choice(self.Delta + 1, p=self.solar_matrix)
        if transmitted:
            next_b = min(max(self.b_t - self.eta + delta_t, 0), self.B)
        else:
            next_b = min(self.b_t + delta_t, self.B)

        #check truncate (whether day is over or not, to end episode)
        ## here terminated is always "False", cuz process never naturally comes to and end (no Game Over, Goal Reached)
        self.time_step += 1
        truncated = self.time_step >= 288
        terminated = False

        #making the internal state update
        self.theta_t = next_theta_t
        self.b_t = next_b
        self.theta_hat = theta_hat_new
        self.theta_max_t = theta_max_new

        ##we're changing up the values, but returning indices for next, so convert to indices
        #converting to idx'es
        next_theta_hat_idx = round(theta_hat_new / 0.02)
        next_theta_max_idx = round(theta_max_new / 0.02)

        next_state = np.array([next_theta_t_idx, next_b, next_theta_hat_idx, next_theta_max_idx])
        return next_state, reward, terminated, truncated, {}
           
    
    def reset(self):
        theta_t_idx = np.random.randint(0, 51)
        theta_hat_idx = np.random.randint(0, 51)
        theta_max_t_idx = theta_t_idx           #this depends on theta_t, cannot be taken randomly, and since we've only seen theta_t upto this point, take this value init to theta+t itself
        b_t_idx = np.random.randint(0, 11)

        self.theta_t = theta_t_idx * 0.02
        self.b_t =  b_t_idx
        self.theta_hat = theta_hat_idx * 0.02
        self.theta_max_t = theta_max_t_idx * 0.02

        self.time_step = 0
        info = {}
        return np.array([theta_t_idx, b_t_idx, theta_hat_idx, theta_max_t_idx]), info 
    
    def render(self):
        # Used to graphics. NOT NEEDED FOR THIS ASSIGNMENT.
        pass