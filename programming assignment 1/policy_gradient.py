import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from RideSharing import DynamicPricingEnv

env = DynamicPricingEnv()

#Feature Eng

def get_closest_driver(passenger_info, driver_info):
    #get closest driver, as highest possibilty of getting picked and reward
    passenger_loc = passenger_info[:2]
    min_dist = float('inf')
    closest_driver = None

    for driver in driver_info:
        driver_loc = driver[:2]

        dist = np.sqrt((passenger_loc[0] - driver_loc[0])**2 + (passenger_loc[1] - driver_loc[1])**2)              #euclidean dist

        if dist < min_dist:
            min_dist = dist
            closest_driver = driver

    return closest_driver

def extract_features(x):
    # context size is variable, so getting a fixed size context vector helps (this is the feature eng part)
    # chose a 10 size vector, passenger_info - 5, closest_driver - 3, min_sensitivity - 1, mean_sensitivity - 1 - (total - 10)
    # sensitivities normalized to 0-1 range
    
    passenger = x[0].copy()
    closest_driver = get_closest_driver(passenger, x[1]).copy()
    all_sens = [d[2] for d in x[1]]
    min_sens = np.min(all_sens)
    mean_sens = np.mean(all_sens)
    
    # normalize sensitivities
    passenger[4] = passenger[4] / 3.0
    closest_driver[2] = closest_driver[2] / 3.0
    min_sens = min_sens / 3.0
    mean_sens = mean_sens / 3.0
    
    feature = np.concatenate([passenger, closest_driver, np.array([min_sens]), np.array([mean_sens])])
    return feature

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 64)
        
        self.layer3 = nn.Linear(64, 1)
        self.eta = nn.Parameter(torch.tensor(0.0)) 
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        mu = self.layer3(x)
        sigma = torch.exp(self.eta)
        return mu, sigma

def init_weights(m):
    #using xavier for stability
    
    if isinstance(m ,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)

def train(episodes=100, batch_size=64):
    baseline = 0
    beta = 0.99
    all_rewards = []

    for episode in range(episodes):
        x, _ = env.reset()
        episode_rewards = []
        episode_features = []
        episode_u_samples = []

        # collecting the whole trajectory
        for t in range(env.Horizon):
            features = extract_features(x)
            features_tensor = torch.FloatTensor(features)
            mu, sigma = model(features_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            u_sample = dist.rsample()
            a = torch.sigmoid(u_sample)
            x, reward, done, truncated, info = env.step(a.detach().item())
            
            # baseline update every step
            baseline = beta * baseline + (1 - beta) * reward
            
            episode_rewards.append(reward)
            episode_features.append(features)
            episode_u_samples.append(u_sample.detach())

        # convert those into tensors
        rewards_tensor = torch.FloatTensor(episode_rewards)
        features_tensor_all = torch.FloatTensor(np.array(episode_features))
        u_samples_tensor = torch.stack(episode_u_samples).squeeze()
        adv = rewards_tensor - baseline

        # mini-batch update as sir said
        indices = torch.randperm(env.Horizon)
        for start in range(0, env.Horizon, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_adv = adv[batch_idx]
            batch_features = features_tensor_all[batch_idx]
            batch_u_samples = u_samples_tensor[batch_idx]

            # recomputing the log probs with stored u
            mu, sigma = model(batch_features)
            dist = torch.distributions.Normal(mu, sigma)
            a = torch.sigmoid(batch_u_samples)
            log_prob = dist.log_prob(batch_u_samples) - torch.log(a * (1 - a) + 1e-6)
            batch_log_probs = log_prob.squeeze()

            loss = -(batch_adv * batch_log_probs).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        all_rewards.extend(episode_rewards)
        print(f"episode {episode+1}/{episodes} | avg reward: {np.mean(episode_rewards):.4f}")

    return all_rewards
def plot_rewards(all_rewards, window = 2000):
    smoothed = []

    for i in range(len(all_rewards)):

        start = max(0, i - window + 1)
        smoothed.append(np.mean(all_rewards[start:i+1]))

    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.xlabel("time step")
    plt.ylabel("receding window avg reward")
    plt.title("policy grad - train")
    plt.grid(True)
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    model = PolicyNetwork()
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)

    all_rewards = train(episodes = 100)
    plot_rewards(all_rewards)

    torch.save(model.state_dict(), 'new_policy_grad_model_temp_100ep_3layer_window2000.pth')

    print(f"train done. final avg rewards = {np.mean(all_rewards[-720*5:]):.4f}")
