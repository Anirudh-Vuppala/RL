import numpy as np
import gymnasium as gym
import pygame

def choose_action():
    # This function should choose action based on the DQN model and the current state.
    # While choosing action here, exploration is not required.
    # You have to set the argumnents of the function and write the required code.
    pass


# The following line load the DQN model. Write (commented) paths for both models, with and without offline data.
# HERE GOES THE LINE.


# The following line initializes the Mountain Car environment with render_mode
# set to 'human'.
# HERE GOES THE LINE.


# The following line resets the environment,
# HERE GOES THE LINE.


end_episode = False
total_reward = 0
while not(end_episode):
    pass # This is a dummy line. REMOVE THIS LINE.
    
    # The following line picks an action using choose_action() function.
    # HERE GOES THE LINE.
    
    
    # The following line takes that the picked action. After taking the action,
    # it gets next state,reward, terminated, truncated, and info.
    # HERE GOES THE LINE.
    
    
    # The following line update the total reward
    # HERE GOES THE LINE.
    
    
    # The following line decides the state for the next time slot.
    # HERE GOES THE LINE.
    
    
    # The following line decides end_episode for the next time slot.
    # HERE GOES THE LINE.


# The following line prints the total reward.
# HERE GOES THE LINE.


# The following line closes the environment.
# HERE GOES THE LINE.

pygame.display.quit()