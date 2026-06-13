import numpy as np
from mdp_utils import *
from Assignment2Tools import prob_vector_generator

# Default parameters
D = 5
S_max = 15
Z_max = 10
tau = 3
lmbda = [1, 0.75, 0.5]

mu_d = 2
stddev_ratio = 0.6
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
phi = prob_vector_generator(D, mu_d, stddev_d)

PERP = get_PERP(D)

print("=" * 50)
print("POLICY VERIFICATION")
print("=" * 50)

print("\n--- No-forecast states (f=[PERP,PERP,PERP]), z=0 ---")
print(f"{'s':>4} | {'d=0':>10} {'d=1':>10} {'d=2':>10} {'d=3':>10} {'d=4':>10} {'d=5':>10}")
print("-" * 70)
for s_test in range(6):
    idx = state_to_index(s_test, 0, [PERP, PERP, PERP], S_max, Z_max, D, tau)
    row = f"{s_test:>4} | "
    for d_test in range(D + 1):
        u, v = policy_optimal_pi[idx, d_test]
        row += f"({u:+d},{v:d})".rjust(10)
    print(row)

print("\n--- No-forecast states (f=[PERP,PERP,PERP]), s=3, varying backlog z ---")
print(f"{'z':>4} | {'d=0':>10} {'d=1':>10} {'d=2':>10} {'d=3':>10} {'d=4':>10} {'d=5':>10}")
print("-" * 70)
for z_test in range(5):
    idx = state_to_index(3, z_test, [PERP, PERP, PERP], S_max, Z_max, D, tau)
    row = f"{z_test:>4} | "
    for d_test in range(D + 1):
        u, v = policy_optimal_pi[idx, d_test]
        row += f"({u:+d},{v:d})".rjust(10)
    print(row)

print("\n--- V values for no-forecast states, z=0 ---")
for s_test in range(6):
    idx = state_to_index(s_test, 0, [PERP, PERP, PERP], S_max, Z_max, D, tau)
    print(f"  s={s_test}, z=0, f=PERP: V = {V_optimal_pi[idx]:.4f}")