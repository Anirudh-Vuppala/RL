import numpy as np
import gymnasium as gym
from GymAirQuality import SensorTransmissionEnv
# --- Helper Functions ---

def get_reward(theta, theta_hat):
    """Calculates the reward (negative loss) based on the assignment formula."""
    diff = abs(theta - theta_hat)
    if theta <= theta_hat:
        return -diff
    else:
        return -1.5 * diff

def get_valid_actions(b_t, eta=2):
    """Returns list of valid actions based on battery constraint."""
    # 0 = Not transmit, 1 = Transmit true, 2 = Transmit max
    if b_t >= eta:
        return [0, 1, 2]
    return [0]

def choose_action(Q, state, epsilon, valid_actions):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        return np.random.choice(valid_actions)
    else:
        theta, b, theta_hat, theta_m = state
        q_values = {a: Q[theta, b, theta_hat, theta_m, a] for a in valid_actions}
        return max(q_values, key=q_values.get)

def testing(env_test, Q, Nep_test=50):
    """Evaluates the greedy policy."""
    avg_reward = 0
    for _ in range(Nep_test):
        state, _ = env_test.reset()
        ep_reward = 0
        done = False
        while not done:
            valid_actions = get_valid_actions(state[1])
            action = choose_action(Q, state, 0.0, valid_actions) 
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        avg_reward += ep_reward
    return avg_reward / Nep_test

# --- Main RL Algorithms ---

def QLearning(env, beta, Nepisodes, alpha):

    # Q-table initialization
    Q = np.zeros((51, 11, 51, 51, 3))

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    test_rewards = []

    # -Training-

    for episode in range(Nepisodes):

        state, _ = env.reset()

        for _ in range(288):

            t, b, th, tm = state

            # use shared function
            valid_actions = get_valid_actions(b)

            # epsilon-greedy selection
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_vals = Q[t, b, th, tm]
                action = valid_actions[np.argmax(q_vals[valid_actions])]

            next_state, reward, terminated, truncated, _ = env.step(action)

            nt, nb, nth, ntm = next_state

            # use shared function
            next_valid_actions = get_valid_actions(nb)

            # max over valid next actions
            max_next_q = max(Q[nt, nb, nth, ntm, a] for a in next_valid_actions)

            # standard Q-learning update
            Q[t, b, th, tm, action] += alpha * (
                reward + beta * max_next_q - Q[t, b, th, tm, action]
            )

            state = next_state

            if truncated:
                break

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # use shared testing function
        if episode % 10 == 0:
            test_rewards.append(testing(env, Q))

    # -Policy extraction-

    policy = np.zeros((51, 11, 51, 51), dtype=int)

    for t in range(51):
        for b in range(11):
            for th in range(51):
                for tm in range(51):

                    valid_actions = get_valid_actions(b)

                    q_vals = Q[t, b, th, tm]
                    best_action = valid_actions[np.argmax(q_vals[valid_actions])]

                    policy[t, b, th, tm] = best_action

    QLearning.rewards = np.array(test_rewards)
    return policy

def QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha, M=10):
    print("Running Structural Knowledge Q-Learning...")
    Q = np.zeros((51, 11, 51, 51, 3))
    B_max = 10
    eta = 2
    
    # ---Initialize tracking arrays and a separate testing environment ---
    reward_test = []
    env_test = SensorTransmissionEnv() 
    
    for ep in range(Nepisodes):
        if ep % 1000 == 0:
            print(f"  Structural Q-Learning: Episode {ep}/{Nepisodes}")
            
        state, _ = env.reset()
        done = False
        epsilon = max(0.01, 1.0 - ep / (Nepisodes * 0.8)) 
        
        while not done:
            theta_1, b_1, theta_hat_1, theta_m_1 = state
            valid_actions = get_valid_actions(b_1, eta)
            a_1 = choose_action(Q, state, epsilon, valid_actions)
            next_state, r_1, terminated, truncated, _ = env.step(a_1)
            done = terminated or truncated
            
            theta_1_prime, b_1_prime, theta_hat_1_prime, theta_m_1_prime = next_state
            
            # 1. Standard Q-Update
            best_next_a = choose_action(Q, next_state, 0.0, get_valid_actions(b_1_prime))
            td_target = r_1 + beta * Q[theta_1_prime, b_1_prime, theta_hat_1_prime, theta_m_1_prime, best_next_a]
            td_error = td_target - Q[theta_1, b_1, theta_hat_1, theta_m_1, a_1]
            Q[theta_1, b_1, theta_hat_1, theta_m_1, a_1] += alpha * td_error
            
            def apply_virtual_update(a_2, r_2, b_2_prime, theta_m_2_prime, theta_hat_2_prime):
                if a_2 in [1, 2] and b_2 < eta:
                    return 
                x_2_prime_best_a = choose_action(Q, [theta_1_prime, b_2_prime, theta_hat_2_prime, theta_m_2_prime], 0.0, get_valid_actions(b_2_prime))
                v_td_target = r_2 + beta * Q[theta_1_prime, b_2_prime, theta_hat_2_prime, theta_m_2_prime, x_2_prime_best_a]
                v_td_error = v_td_target - Q[theta_1, b_2, theta_hat_2, theta_m_2, a_2]
                Q[theta_1, b_2, theta_hat_2, theta_m_2, a_2] += alpha * v_td_error

            # 2. Structural Updates (Virtual Experience)
            if b_1_prime < B_max:

                if a_1 == 0:
                    for _ in range(5):
                        theta_m_2 = np.random.randint(0, 51)
                        theta_hat_2 = np.random.randint(0, 51)
                        b_2 = np.random.randint(0, 11)
                        r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                        b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                        tm_2_p = max(theta_m_2, theta_1_prime)
                        apply_virtual_update(0, r_2, b_2_p, tm_2_p, theta_hat_2)

                elif a_1 == 1: 
                    for _ in range(5): 
                        theta_m_2 = np.random.randint(0, 51)
                        theta_hat_2 = np.random.randint(0, 51)
                        b_2 = np.random.randint(0, 11)
                        
                        r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                        b_2_p = min(b_2 + (b_1_prime - b_1 + eta), B_max)
                        tm_2_p = max(theta_m_2, theta_1_prime)
                        apply_virtual_update(0, r_2, b_2_p, tm_2_p, theta_hat_2)

                        if theta_1 != theta_hat_1:
                            if theta_hat_1_prime == theta_hat_1: 
                                r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = max(theta_m_2, theta_1_prime)
                                apply_virtual_update(1, r_2, b_2_p, tm_2_p, theta_hat_2)
                            elif theta_hat_1_prime == theta_1:   
                                r_2 = 0
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = theta_1_prime
                                apply_virtual_update(1, r_2, b_2_p, tm_2_p, theta_1)

                        if theta_1 != theta_hat_1:
                            if theta_hat_1_prime == theta_hat_1: 
                                r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = max(theta_m_2, theta_1_prime)
                                apply_virtual_update(2, r_2, b_2_p, tm_2_p, theta_hat_2)
                            elif theta_hat_1_prime == theta_1:   
                                r_2 = get_reward(theta_1 * 0.02, theta_m_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = theta_1_prime
                                apply_virtual_update(2, r_2, b_2_p, tm_2_p, theta_m_2)

                elif a_1 == 2:
                    for _ in range(5): 
                        theta_m_2 = np.random.randint(0, 51)
                        theta_hat_2 = np.random.randint(0, 51)
                        b_2 = np.random.randint(0, 11)

                        r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                        b_2_p = min(b_2 + (b_1_prime - b_1 + eta), B_max)
                        tm_2_p = max(theta_m_2, theta_1_prime)
                        apply_virtual_update(0, r_2, b_2_p, tm_2_p, theta_hat_2)

                        if theta_hat_1 != theta_m_1:
                            if theta_hat_1_prime == theta_hat_1: 
                                r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = max(theta_m_2, theta_1_prime)
                                apply_virtual_update(1, r_2, b_2_p, tm_2_p, theta_hat_2)
                            elif theta_hat_1_prime == theta_m_1: 
                                r_2 = 0
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = theta_1_prime
                                apply_virtual_update(1, r_2, b_2_p, tm_2_p, theta_1)

                        if theta_hat_1 != theta_m_1:
                            if theta_hat_1_prime == theta_hat_1: 
                                r_2 = get_reward(theta_1 * 0.02, theta_hat_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = max(theta_m_2, theta_1_prime)
                                apply_virtual_update(2, r_2, b_2_p, tm_2_p, theta_hat_2)
                            elif theta_hat_1_prime == theta_m_1: 
                                r_2 = get_reward(theta_1 * 0.02, theta_m_2 * 0.02)
                                b_2_p = min(b_2 + (b_1_prime - b_1), B_max)
                                tm_2_p = theta_1_prime
                                apply_virtual_update(2, r_2, b_2_p, tm_2_p, theta_m_2)

            state = next_state
        
        # ---Run testing every M episodes and append to tracking list ---
        if ep % M == 0:
            r_avg = testing(env_test, Q, Nep_test=50)
            reward_test.append(r_avg)

    # Convert the trained Q-table into a deterministic policy
    policy = np.zeros((51, 11, 51, 51), dtype=int)
    for t in range(51):
        for b in range(11):
            for th in range(51):
                for tm in range(51):
                    valid_acts = get_valid_actions(b)
                    q_vals = {a: Q[t, b, th, tm, a] for a in valid_acts}
                    policy[t, b, th, tm] = max(q_vals, key=q_vals.get)
    
    # Save the learning curve data so you can plot it later for your report
    np.save('reward_test.npy', np.array(reward_test))
                    
    return policy

print("Initializing environment...")
env = SensorTransmissionEnv()

Nepisodes = 10000
alpha = 0.1
beta = 0.98

print("Training policies (this will take a few minutes)...")
policy2 = QLearning_StructuralKnowledge(env, beta, Nepisodes, alpha)
policy1 = QLearning(env, beta, Nepisodes, alpha)

print("Saving policy...")

np.save('policy1.npy', policy1)
np.save('policy2.npy', policy2)

# --- Separate post-training evaluation ---
# Load the saved policy and run a fresh episodic loop to collect reward data.
print("\nStarting post-training evaluation...")
policy2_loaded = np.load('policy2.npy')
env_eval = SensorTransmissionEnv()
Nep_eval = 500
rewards2 = []

for _ in range(Nep_eval):
    state, _ = env_eval.reset()
    ep_reward = 0
    done = False
    while not done:
        theta, b, theta_hat, theta_m = state
        valid_actions = get_valid_actions(b)
        action = int(policy2_loaded[theta, b, theta_hat, theta_m])
        if action not in valid_actions:
            action = 0
        next_state, reward, terminated, truncated, _ = env_eval.step(action)
        done = terminated or truncated
        ep_reward += reward
        state = next_state
    rewards2.append(ep_reward)

np.save('rewards1.npy', QLearning.rewards)
np.save('rewards2.npy', np.array(rewards2))
print("Saved 'rewards2.npy'.")

env.close()
