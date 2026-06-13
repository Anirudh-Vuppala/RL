import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  DQN Architecture 1
#  Input : [position, velocity, one_hot(action)]  → size 5
#  Output: scalar Q-value q(x, a)
# ─────────────────────────────────────────────
class DQN_Arch1(nn.Module):
    def __init__(self):
        super(DQN_Arch1, self).__init__()
        # Input size = 2 (state) + 3 (one-hot action) = 5
        # Two hidden layers, 64 neurons each → model stays well under 100 KB
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # single scalar output: Q(x, a)
        )

    def forward(self, x):
        # x shape: (batch, 5)  →  output shape: (batch,)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
#  Helper: one-hot encode a single integer action
#  e.g. action=0 → [1,0,0], action=2 → [0,0,1]
# ─────────────────────────────────────────────
def one_hot(action, n_actions=3):
    oh = np.zeros(n_actions, dtype=np.float32)
    oh[action] = 1.0
    return oh


# ─────────────────────────────────────────────
#  Helper: build a batched input tensor for Architecture 1
#  states  : np.array (N, 2)
#  actions : np.array (N,)  integer actions
#  returns : torch.FloatTensor (N, 5)
# ─────────────────────────────────────────────
def make_input(states, actions, n_actions=3):
    oh = np.eye(n_actions, dtype=np.float32)[actions]   # (N, 3)
    x  = np.concatenate([states, oh], axis=1)            # (N, 5)
    return torch.FloatTensor(x)


# ─────────────────────────────────────────────
#  Custom exploration strategy (from professor's YT video, Rule 2)
#
#  With prob ε:
#    1. Compute Q(x,a) for all actions
#    2. Normalise: q̄(x,a) = q(x,a) / (υ + Σ_ã|q(x,ã)|)
#    3. Probabilities: P_a = exp(q̄(x,a)) / Σ_ã exp(q̄(x,ã))
#    4. Sample action ~ P_a
#
#  With prob (1-ε):
#    a = argmax_ã Q(x, ã)
# ─────────────────────────────────────────────
def choose_action(state, model_predict, epsilon, n_actions=3):
    """
    state        : np.array (2,)
    model_predict: DQN_Arch1
    epsilon      : float, current exploration probability
    """
    model_predict.eval()
    with torch.no_grad():
        # Forward pass for every action to get Q(x, a) for all a
        q_vals = np.array([
            model_predict(
                torch.FloatTensor(np.concatenate([state, one_hot(a)])).unsqueeze(0)
            ).item()
            for a in range(n_actions)
        ])  # shape: (3,)

    if np.random.uniform() < epsilon:
        # ── Exploration: softmax over normalised Q-values ──────────────
        # Normalise Q-values (absolute value only in denominator, per slide)
        upsilon = 1e-8
        denom   = upsilon + np.sum(np.abs(q_vals))
        q_norm  = q_vals / denom

        # Softmax probabilities (subtract max for numerical stability)
        exp_q = np.exp(q_norm - np.max(q_norm))
        probs = exp_q / np.sum(exp_q)

        action = np.random.choice(n_actions, p=probs)
    else:
        # ── Exploitation: greedy argmax ────────────────────────────────
        action = int(np.argmax(q_vals))

    return action


# ─────────────────────────────────────────────
#  Generate additional state-action pairs (Rule 3 from YT video)
#
#  Allowed knowledge:
#    position_{t+1} = position_t + velocity_{t+1}   (position update eq.)
#    reward = -1 always, 0 if goal reached (position >= 0.45)
#
#  NOT allowed:
#    velocity_{t+1} = velocity_t + (action-1)*force - cos(3*pos_t)*gravity
#
#  Strategy (pseudocode for report.pdf):
#  -----------------------------------------------------------------------
#  For each sample (x, a, r, x') in the sampled batch:
#    pos, vel   = x
#    pos', vel' = x'
#    # vel' is already observed in x', so we know it without the velocity eq.
#    For k = 1 to K:
#      δ        ~ Uniform(-0.05, 0.05)
#      pos_alt  = clip(pos + δ,  -1.2, 0.6)
#      pos_alt' = clip(pos_alt + vel',  -1.2, 0.6)   # position update eq.
#      r_alt    = 0.0 if pos_alt' >= 0.45 else -1.0  # reward knowledge
#      term_alt = 1   if r_alt == 0.0   else 0
#      Append (x_alt=[pos_alt, vel], a, r_alt, x_alt'=[pos_alt', vel'], term_alt)
#  Concatenate extra samples with original batch → augmented batch
#  -----------------------------------------------------------------------
# ─────────────────────────────────────────────
GOAL_POSITION = 0.45
K_SYNTHETIC   = 2      # extra samples generated per real sample

def generate_additional_samples(s_b, a_b, r_b, ns_b, t_b):
    """
    s_b, ns_b : np.array (N, 2)
    a_b       : np.array (N,)  int
    r_b       : np.array (N,)  float
    t_b       : np.array (N,)  int
    Returns augmented versions of all five arrays.
    """
    extra_s, extra_a, extra_r, extra_ns, extra_t = [], [], [], [], []

    for i in range(len(s_b)):
        pos      = s_b[i, 0]
        vel      = s_b[i, 1]
        vel_next = ns_b[i, 1]   # observed next velocity (no velocity eq. needed)
        a        = a_b[i]

        for _ in range(K_SYNTHETIC):
            delta        = np.random.uniform(-0.05, 0.05)
            pos_alt      = np.clip(pos + delta,        -1.2, 0.6)
            pos_alt_next = np.clip(pos_alt + vel_next, -1.2, 0.6)

            r_alt = 0.0 if pos_alt_next >= GOAL_POSITION else -1.0
            term  = 1   if r_alt == 0.0 else 0

            extra_s.append([pos_alt,      vel])
            extra_a.append(a)
            extra_r.append(r_alt)
            extra_ns.append([pos_alt_next, vel_next])
            extra_t.append(term)

    extra_s  = np.array(extra_s,  dtype=np.float32)
    extra_a  = np.array(extra_a,  dtype=np.int64)
    extra_r  = np.array(extra_r,  dtype=np.float32)
    extra_ns = np.array(extra_ns, dtype=np.float32)
    extra_t  = np.array(extra_t,  dtype=np.int64)

    return (np.concatenate([s_b,  extra_s],  axis=0),
            np.concatenate([a_b,  extra_a],  axis=0),
            np.concatenate([r_b,  extra_r],  axis=0),
            np.concatenate([ns_b, extra_ns], axis=0),
            np.concatenate([t_b,  extra_t],  axis=0))


# ─────────────────────────────────────────────
#  Epsilon scheduler: linear decay 1.0 → 0.05
# ─────────────────────────────────────────────
def exploration_prob_scheduler(episode, total_episodes):
    epsilon_start = 1.0
    epsilon_end   = 0.05
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / total_episodes)
    return float(np.clip(epsilon, epsilon_end, epsilon_start))


# ─────────────────────────────────────────────
#  load_offline_data  (provided by professor — DO NOT MODIFY)
# ─────────────────────────────────────────────
def load_offline_data(path, min_score):
    state_data, action_data, reward_data = [], [], []
    next_state_data, terminated_data     = [], []

    dataset       = pd.read_csv(path)
    dataset_group = dataset.groupby('Episode #')

    for play_no, df in dataset_group:
        start_idx = 0
        if isinstance(df.iloc[0, 1], str) and '{}' in df.iloc[0, 1]:
            start_idx = 1
        df = df[start_idx:]

        state = []
        for s in df.iloc[:, 1]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                state.append([float(v.strip(',')) for v in s])
            else:
                state.append(s)
        state = np.array(state)

        action = np.array(df.iloc[:, 2]).astype(int)
        reward = np.array(df.iloc[:, 3]).astype(np.float32)

        next_state = []
        for s in df.iloc[:, 4]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                next_state.append([float(v.strip(',')) for v in s])
            else:
                next_state.append(s)
        next_state = np.array(next_state)

        terminated   = np.array(df.iloc[:, 5]).astype(int)
        total_reward = np.sum(reward)

        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)

    if not state_data:
        return (np.array([]),) * 5

    return (np.concatenate(state_data),
            np.concatenate(action_data),
            np.concatenate(reward_data),
            np.concatenate(next_state_data),
            np.concatenate(terminated_data))


# ─────────────────────────────────────────────
#  plot_reward
# ─────────────────────────────────────────────
def plot_reward(total_reward_per_episode, window_length):
    """
    Plots (i) total reward per episode and
          (ii) moving average (window slides by 1 episode each time)
    on the same graph.
    """
    rewards  = np.array(total_reward_per_episode)
    episodes = np.arange(1, len(rewards) + 1)

    # Moving average: episode i → mean of last window_length episodes
    moving_avg = np.array([
        np.mean(rewards[max(0, i - window_length):i])
        for i in range(1, len(rewards) + 1)
    ])

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards,    alpha=0.4,   color='steelblue',
             label='Total reward per episode')
    plt.plot(episodes, moving_avg, linewidth=2, color='darkorange',
             label=f'Moving average (window={window_length})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Double DQN Training – Mountain Car')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_plot.png', dpi=150)
    plt.show()
    print('Reward plot saved to reward_plot.png')


# ─────────────────────────────────────────────
#  DQN_training  –  Double DQN, Algorithm 5, Architecture 1
# ─────────────────────────────────────────────
def DQN_training(env, offline_data, use_offline_data):
    """
    Returns
    -------
    model_predict            : trained PyTorch predict DQN (Architecture 1)
    total_reward_per_episode : np.array of per-episode total rewards
    """

    # ── Hyperparameters (Algorithm 5, line 3) ─────────────────────────
    N_ACTIONS   = 3
    N_OBS       = 2
    Nepisodes   = 5000      # total episodes
    Nu          = 4         # predict DQN update period (every Nu timesteps)
    Nt          = 200       # target DQN update period  (every Nt timesteps)
    Nb          = 64        # training batch size
    beta        = 0.99      # discount factor
    alpha       = 1e-3      # learning rate
    buffer_size = 50000     # replay buffer capacity
    Nsave       = 500       # checkpoint every Nsave timesteps
    E           = 50        # freeze env-data for first E episodes (offline mode)

    # ── Line 1: Initialise predict and target DQNs ────────────────────
    model_predict = DQN_Arch1()
    model_target  = DQN_Arch1()
    model_target.load_state_dict(model_predict.state_dict())
    model_target.eval()   # target net is never directly trained

    optimizer = optim.Adam(model_predict.parameters(), lr=alpha)
    loss_fn   = nn.MSELoss()

    # ── Line 2: Initialise replay buffer ──────────────────────────────
    state_buf  = np.zeros((buffer_size, N_OBS), dtype=np.float32)
    action_buf = np.zeros(buffer_size,           dtype=np.int64)
    reward_buf = np.zeros(buffer_size,           dtype=np.float32)
    next_buf   = np.zeros((buffer_size, N_OBS), dtype=np.float32)
    term_buf   = np.zeros(buffer_size,           dtype=np.int64)

    buf_count = 0   # number of valid entries currently in buffer
    buf_ix    = 0   # next write index (cyclic)

    # Pre-fill buffer with offline human data (use_offline_data=True only)
    if use_offline_data:
        od_s, od_a, od_r, od_ns, od_t = offline_data
        if len(od_s) > 0:
            n_off = min(len(od_s), buffer_size)
            for i in range(n_off):
                state_buf[buf_ix]  = od_s[i]
                action_buf[buf_ix] = od_a[i]
                reward_buf[buf_ix] = od_r[i]
                next_buf[buf_ix]   = od_ns[i]
                term_buf[buf_ix]   = od_t[i]
                buf_count = min(buf_count + 1, buffer_size)
                buf_ix    = (buf_ix + 1) % buffer_size
            print(f'Buffer pre-filled with {n_off} offline samples.')

    # ── Line 3: counter = 0 ───────────────────────────────────────────
    # Single counter exactly as in Algorithm 5 — used for BOTH Nu and Nt checks
    counter = 0

    total_reward_per_episode = []

    # ── Line 4: Episode loop ──────────────────────────────────────────
    for episode in range(Nepisodes):

        # Line 5: Reset environment
        x, _ = env.reset()

        # Line 6: Choose ε for this episode
        epsilon = exploration_prob_scheduler(episode, Nepisodes)

        total_reward = 0.0
        end_episode  = False

        # Freeze adding env samples to buffer during offline warm-up period
        freeze_buffer = use_offline_data and (episode < E)

        # ── Line 7: Timestep loop ─────────────────────────────────────
        while not end_episode:

            # Line 8: Pick action using custom exploration strategy
            a = choose_action(x, model_predict, epsilon, N_ACTIONS)

            # Line 9: Take action
            x_dash, r, terminated, truncated, _ = env.step(a)
            total_reward += r

            # Line 10: Append (x, a, r, x') to replay buffer
            if not freeze_buffer:
                state_buf[buf_ix]  = x
                action_buf[buf_ix] = a
                reward_buf[buf_ix] = r
                # Store zeros for next state if terminal (no future state exists)
                next_buf[buf_ix]   = x_dash if not terminated else np.zeros(N_OBS)
                term_buf[buf_ix]   = int(terminated)
                buf_count = min(buf_count + 1, buffer_size)
                buf_ix    = (buf_ix + 1) % buffer_size

            # Line 11: if counter % Nu == 0 → update predict DQN
            if counter % Nu == 0 and buf_count >= Nb:

                # Line 12: Sample random batch of size Nb
                ix   = np.random.choice(buf_count, size=Nb, replace=False)
                s_b  = state_buf[ix].copy()
                a_b  = action_buf[ix].copy()
                r_b  = reward_buf[ix].copy()
                ns_b = next_buf[ix].copy()
                t_b  = term_buf[ix].copy()

                # Rule 3: augment batch with synthetic state-action pairs
                s_b, a_b, r_b, ns_b, t_b = generate_additional_samples(
                    s_b, a_b, r_b, ns_b, t_b)

                aug_size = len(s_b)   # = Nb * (1 + K_SYNTHETIC)

                # Line 13: Compute targets y using Double DQN rule
                #   a* = argmax_{a'} q̂(x', a'; φ_P)  ← predict net selects action
                #   y  = r + β * q̂(x', a*; φ_T)      ← target net evaluates it
                model_predict.eval()
                with torch.no_grad():
                    # Q(x', all 3 actions) via predict net → find a* per sample
                    ns_rep   = np.repeat(ns_b, N_ACTIONS, axis=0)         # (aug*3, 2)
                    all_acts = np.tile(np.arange(N_ACTIONS), aug_size)    # (aug*3,)
                    q_p_next = model_predict(make_input(ns_rep, all_acts)) # (aug*3,)
                    q_p_next = q_p_next.numpy().reshape(aug_size, N_ACTIONS)  # (aug, 3)
                    a_star   = np.argmax(q_p_next, axis=1)                # (aug,)

                    # Q(x', a*) via target net  — uses AUGMENTED ns_b
                    q_t_next = model_target(make_input(ns_b, a_star))     # (aug,)
                    q_t_next = q_t_next.numpy()

                    # Bellman target: (1 - term) zeros out future reward at terminal states
                    targets  = r_b + beta * q_t_next * (1 - t_b)         # (aug,)

                # Line 14: One gradient descent step on predict DQN
                model_predict.train()
                q_pred   = model_predict(make_input(s_b, a_b))  # (aug,)
                y_target = torch.FloatTensor(targets)            # (aug,)

                loss = loss_fn(q_pred, y_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Line 15: if counter % Nt == 0 → copy weights to target net
            if counter % Nt == 0:
                model_target.load_state_dict(model_predict.state_dict())

            # Periodic checkpoint
            if counter % Nsave == 0 and counter > 0:
                save_name = 'DQN_offline_true' if use_offline_data else 'DQN_offline_false'
                torch.save(model_predict.state_dict(), f'{save_name}_ckpt.pth')

            # Line 17: x ← x',  counter ← counter + 1
            x       = np.copy(x_dash)
            counter += 1

            if terminated or truncated:
                end_episode = True

        total_reward_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(total_reward_per_episode[-100:])
            print(f'Episode={episode+1:5d} | Reward={total_reward:7.1f} | '
                  f'Avg(100)={recent_avg:7.1f} | ε={epsilon:.3f} | '
                  f'Buffer={buf_count:6d} | Steps={counter}')

    return model_predict, np.array(total_reward_per_episode)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

# Initiate the mountain car environment.
# NO RENDERING. It will slow the training process.
env = gym.make('MountainCar-v0')

# Load the offline data collected in step 3. Also, process the dataset.
path      = 'car_dataset.csv'
min_score = -np.inf
offline_data = load_offline_data(path, min_score)

# Train DQN model
# Set use_offline_data=False for your task, True for Vivi's task
use_offline_data = False
final_model, total_reward_per_episode = DQN_training(env, offline_data, use_offline_data)

# Save the final model (PyTorch)
# DQN_offline_false → trained without offline data  (your task)
# DQN_offline_true  → trained with offline data     (Vivi's task)
model_name = 'DQN_offline_true' if use_offline_data else 'DQN_offline_false'
torch.save(final_model.state_dict(), f'{model_name}.pth')
print(f'Model saved as {model_name}.pth')

# Plot reward per episode and moving average reward
window_length = 50
plot_reward(total_reward_per_episode, window_length)

env.close()