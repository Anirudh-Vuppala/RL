import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──
rewards2 = np.load('reward_test.npy')
policy2 = np.load('policy2.npy')

# ══════════════════════════════════════════
# DELIVERABLE 1: Learning Curve Plot
# ══════════════════════════════════════════
def smooth(data, window=30):
    return np.convolve(data, np.ones(window)/window, mode='valid')

x2 = np.arange(0, len(rewards2) * 10, 10)
r2_smooth = smooth(rewards2)
x2_smooth = x2[:len(r2_smooth)]

plt.figure(figsize=(10, 6))
plt.plot(x2, rewards2, alpha=0.2, color='orange', label='SK Q-Learning (raw)')
plt.plot(x2_smooth, r2_smooth, color='orange', linewidth=2,
         label='SK Q-Learning (smoothed)')
plt.xlabel('Episode')
plt.ylabel('Average Sum of Reward (over 50 test episodes)')
plt.title('Learning Curve: Structural Knowledge Q-Learning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.show()
print(f"Final converged reward: {rewards2[-50:].mean():.4f}")

# ══════════════════════════════════════════
# COLLECT STATE DATA PER ACTION
# ══════════════════════════════════════════

# Prof's Action 1 = our action 0 (Don't transmit)
action1_battery = []
action1_error = []
action1_theta = []

# Prof's Action 2 = our action 1 (Transmit true θ)
action2_battery = []
action2_error = []
action2_theta = []
action2_max_minus_true = []

for t in range(51):
    for b in range(11):
        for th in range(51):
            for tm in range(51):
                action = policy2[t, b, th, tm]
                theta = t * 0.02
                theta_hat = th * 0.02
                theta_max = tm * 0.02
                error = theta - theta_hat

                if action == 0:  # Prof's Action 1: Don't transmit
                    action1_battery.append(b)
                    action1_error.append(error)
                    action1_theta.append(theta)

                elif action == 1:  # Prof's Action 2: Transmit true θ
                    action2_battery.append(b)
                    action2_error.append(error)
                    action2_theta.append(theta)
                    action2_max_minus_true.append(theta_max - theta)

action1_battery = np.array(action1_battery)
action1_error = np.array(action1_error)
action1_theta = np.array(action1_theta)

action2_battery = np.array(action2_battery)
action2_error = np.array(action2_error)
action2_theta = np.array(action2_theta)
action2_max_minus_true = np.array(action2_max_minus_true)

# ══════════════════════════════════════════
# DELIVERABLE 2: Action 1 (Don't Transmit)
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(action1_battery, bins=11, range=(0,10),
             color='steelblue', edgecolor='black')
axes[0].set_xlabel('Battery Level (b)')
axes[0].set_ylabel('Count')
axes[0].set_title('Battery Level')
axes[0].grid(True, alpha=0.3)

axes[1].hist(action1_error, bins=30, color='steelblue', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', label='Zero error')
axes[1].set_xlabel('θ - θ̂ (Estimation Error)')
axes[1].set_ylabel('Count')
axes[1].set_title('Estimation Error (θ - θ̂)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(action1_theta, bins=20, color='steelblue', edgecolor='black')
axes[2].set_xlabel('True Pollution θ')
axes[2].set_ylabel('Count')
axes[2].set_title('True Pollution Distribution')
axes[2].grid(True, alpha=0.3)

plt.suptitle('States where Action 1 (Do Not Transmit) is Optimal',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('action1_analysis.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# DELIVERABLE 3: Action 2 (Transmit True θ)
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(action2_battery, bins=11, range=(0,10),
             color='darkorange', edgecolor='black')
axes[0].set_xlabel('Battery Level (b)')
axes[0].set_ylabel('Count')
axes[0].set_title('Battery Level')
axes[0].grid(True, alpha=0.3)

axes[1].hist(action2_error, bins=30, color='darkorange', edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', label='Zero error')
axes[1].set_xlabel('θ - θ̂ (Estimation Error)')
axes[1].set_ylabel('Count')
axes[1].set_title('Estimation Error (θ - θ̂)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(action2_max_minus_true, bins=20, color='darkorange', edgecolor='black')
axes[2].set_xlabel('θ_max - θ')
axes[2].set_ylabel('Count')
axes[2].set_title('How much θ_max exceeds θ')
axes[2].grid(True, alpha=0.3)

plt.suptitle('States where Action 2 (Transmit True θ) is Optimal',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('action2_analysis.png', dpi=150)
plt.show()

# ══════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════
print("\n── ACTION 1 SUMMARY (Don't Transmit) ──")
print(f"Total states: {len(action1_battery)}")
print(f"Average battery: {action1_battery.mean():.2f}")
print(f"Average error (θ-θ̂): {action1_error.mean():.4f}")
print(f"% underestimation (θ > θ̂): {(action1_error > 0).mean()*100:.1f}%")
print(f"% overestimation (θ < θ̂): {(action1_error < 0).mean()*100:.1f}%")

print("\n── ACTION 2 SUMMARY (Transmit True θ) ──")
print(f"Total states: {len(action2_battery)}")
print(f"Average battery: {action2_battery.mean():.2f}")
print(f"Average error (θ-θ̂): {action2_error.mean():.4f}")
print(f"% underestimation (θ > θ̂): {(action2_error > 0).mean()*100:.1f}%")
print(f"% overestimation (θ < θ̂): {(action2_error < 0).mean()*100:.1f}%")
print(f"Average θ_max - θ: {action2_max_minus_true.mean():.4f}")