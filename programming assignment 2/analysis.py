import numpy as np
import matplotlib.pyplot as plt
import time
from mdp_utils import *
from Assignment2Tools import prob_vector_generator
from policy_iteration import policy_iteration

# =============================================================================
# BASE PARAMETERS (default from assignment)
# =============================================================================
D = 5
S_max = 15
theta_base = [5, 1, 0.25]
mu_d = 2
stddev_ratio = 0.6
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
phi_base = prob_vector_generator(D, mu_d, stddev_d)
Z_max = 10
alpha = 0.5
tau_base = 3
lmbda_base = [1, 0.75, 0.5]
beta = 0.95
threshold = 0.1
Kmin = 10

PERP = get_PERP(D)


def count_switching(policy, S_max, Z_max, D, tau):
    """
    For all no-forecast states (f=[PERP]*tau), count fraction of
    (state, demand) pairs where u > 0 (switch ON) and u < 0 (switch OFF).
    """
    switch_on = 0
    switch_off = 0
    total = 0
    for s in range(S_max + 1):
        for z in range(Z_max + 1):
            f = [PERP] * tau
            idx = state_to_index(s, z, f, S_max, Z_max, D, tau)
            for d in range(D + 1):
                u, v = policy[idx, d]
                if u > 0:
                    switch_on += 1
                elif u < 0:
                    switch_off += 1
                total += 1
    return switch_on / total, switch_off / total


def avg_utilization_gap(policy, S_max, Z_max, D, tau):
    """
    Average gap (servers_on - v) across all no-forecast states with d > 0.
    Measures under-utilization: servers available but not used.
    """
    gaps = []
    for s in range(S_max + 1):
        for z in range(Z_max + 1):
            f = [PERP] * tau
            idx = state_to_index(s, z, f, S_max, Z_max, D, tau)
            for d in range(1, D + 1):  # skip d=0, no demand to serve
                u, v = policy[idx, d]
                servers_on = s + u
                gap = servers_on - v
                gaps.append(gap)
    return np.mean(gaps)


def ref_V(V, S_max, Z_max, D, tau):
    """V at reference state s=3, z=0, f=PERP."""
    idx = state_to_index(3, 0, [PERP] * tau, S_max, Z_max, D, tau)
    return V[idx]


# =============================================================================
# TASK 1 — Claim 1: Deferral reduces switching ON
# Fix tau=0. Vary Z_max. More deferral -> less switching ON.
# =============================================================================
print("\n" + "=" * 60)
print("TASK 1 — Claim 1: Deferral reduces switching ON (tau=0)")
print("=" * 60)

task1_results = []
for Z_test in [0, 2, 5, 10]:
    print(f"\n  Running Z_max={Z_test}...")
    t0 = time.time()
    V, pol = policy_iteration(
        D, S_max, Z_test, 0, phi_base, [], theta_base,
        alpha, beta, threshold, Kmin
    )
    frac_on, frac_off = count_switching(pol, S_max, Z_test, D, 0)
    v_ref = ref_V(V, S_max, Z_test, D, 0)
    task1_results.append((Z_test, frac_on, frac_off, v_ref))
    print(f"  Z_max={Z_test} | Switch ON: {frac_on:.3f} | "
          f"Switch OFF: {frac_off:.3f} | V(ref): {v_ref:.4f} "
          f"| Time: {time.time()-t0:.1f}s")

print("\n--- Task 1 Summary ---")
print(f"{'Z_max':>8} {'Switch ON frac':>16} {'Switch OFF frac':>16} {'V(s=3,z=0)':>12}")
print("-" * 56)
for Z_test, fon, foff, vr in task1_results:
    print(f"{Z_test:>8} {fon:>16.3f} {foff:>16.3f} {vr:>12.4f}")


# =============================================================================
# TASK 2 — Claim 2: Forecasting reduces switching OFF
# Fix Z_max=10. Vary tau. More forecasting -> less switching OFF.
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2 — Claim 2: Forecasting reduces switching OFF")
print("=" * 60)

task2_results = []
for tau_test in [0, 1, 2, 3]:
    lmbda_test = [1, 0.75, 0.5][:tau_test]
    print(f"\n  Running tau={tau_test}...")
    t0 = time.time()
    V, pol = policy_iteration(
        D, S_max, Z_max, tau_test, phi_base, lmbda_test, theta_base,
        alpha, beta, threshold, Kmin
    )
    frac_on, frac_off = count_switching(pol, S_max, Z_max, D, tau_test)
    v_ref = ref_V(V, S_max, Z_max, D, tau_test)
    task2_results.append((tau_test, frac_on, frac_off, v_ref))
    print(f"  tau={tau_test} | Switch ON: {frac_on:.3f} | "
          f"Switch OFF: {frac_off:.3f} | V(ref): {v_ref:.4f} "
          f"| Time: {time.time()-t0:.1f}s")

print("\n--- Task 2 Summary ---")
print(f"{'tau':>6} {'Switch ON frac':>16} {'Switch OFF frac':>16} {'V(s=3,z=0)':>12}")
print("-" * 54)
for tau_test, fon, foff, vr in task2_results:
    print(f"{tau_test:>6} {fon:>16.3f} {foff:>16.3f} {vr:>12.4f}")


# =============================================================================
# TASK 3 — Deferral benefit increases with demand variance (tau=0)
# Vary stddev_ratio. Compare V* with Z_max=0 vs Z_max=10.
# Gap should grow as variance increases.
# =============================================================================
print("\n" + "=" * 60)
print("TASK 3 — Deferral benefit vs demand variance (tau=0)")
print("=" * 60)

task3_results = []
for sr in [0.1, 0.3, 0.6, 0.9]:
    sd = sr * np.sqrt(D * (D - mu_d))
    phi_test = prob_vector_generator(D, mu_d, sd)
    print(f"\n  Running stddev_ratio={sr}...")

    t0 = time.time()
    V0, _ = policy_iteration(
        D, S_max, 0, 0, phi_test, [], theta_base,
        alpha, beta, threshold, Kmin
    )
    v_no_defer = ref_V(V0, S_max, 0, D, 0)

    V10, _ = policy_iteration(
        D, S_max, Z_max, 0, phi_test, [], theta_base,
        alpha, beta, threshold, Kmin
    )
    v_defer = ref_V(V10, S_max, Z_max, D, 0)

    gap = v_no_defer - v_defer
    task3_results.append((sr, v_no_defer, v_defer, gap))
    print(f"  stddev={sr} | V(no defer): {v_no_defer:.4f} | "
          f"V(defer): {v_defer:.4f} | Gap: {gap:.4f} "
          f"| Time: {time.time()-t0:.1f}s")

print("\n--- Task 3 Summary ---")
print(f"{'stddev_ratio':>14} {'V(Z_max=0)':>12} {'V(Z_max=10)':>13} {'Gap':>10}")
print("-" * 52)
for sr, v0, v10, gap in task3_results:
    print(f"{sr:>14} {v0:>12.4f} {v10:>13.4f} {gap:>10.4f}")


# =============================================================================
# TASK 4 — theta3 causes under-utilization (tau=0)
# Vary theta3. Average gap (servers_on - v) should increase with theta3.
# =============================================================================
print("\n" + "=" * 60)
print("TASK 4 — theta3 and server under-utilization (tau=0)")
print("=" * 60)

task4_results = []
for theta3_test in [0.1, 0.25, 0.5, 1.0, 2.0]:
    theta_test = [5, 1, theta3_test]
    print(f"\n  Running theta3={theta3_test}...")
    t0 = time.time()
    V, pol = policy_iteration(
        D, S_max, Z_max, 0, phi_base, [], theta_test,
        alpha, beta, threshold, Kmin
    )
    avg_gap = avg_utilization_gap(pol, S_max, Z_max, D, 0)
    v_ref = ref_V(V, S_max, Z_max, D, 0)
    task4_results.append((theta3_test, avg_gap, v_ref))
    print(f"  theta3={theta3_test} | Avg idle servers: {avg_gap:.3f} | "
          f"V(ref): {v_ref:.4f} | Time: {time.time()-t0:.1f}s")

print("\n--- Task 4 Summary ---")
print(f"{'theta3':>8} {'Avg idle servers':>18} {'V(s=3,z=0)':>12}")
print("-" * 42)
for theta3_test, avg_gap, vr in task4_results:
    print(f"{theta3_test:>8} {avg_gap:>18.3f} {vr:>12.4f}")


# =============================================================================
# PLOTS — save as figures for report
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Section 7 Analysis", fontsize=14)

# Task 1
ax = axes[0, 0]
z_vals = [r[0] for r in task1_results]
fon_vals = [r[1] for r in task1_results]
ax.plot(z_vals, fon_vals, 'o-', color='steelblue')
ax.set_xlabel("Z_max (deferral capacity)")
ax.set_ylabel("Fraction of states with u > 0")
ax.set_title("Task 1: Deferral vs Switching ON")
ax.grid(True, alpha=0.3)

# Task 2
ax = axes[0, 1]
tau_vals = [r[0] for r in task2_results]
foff_vals = [r[2] for r in task2_results]
ax.plot(tau_vals, foff_vals, 'o-', color='coral')
ax.set_xlabel("tau (forecast window)")
ax.set_ylabel("Fraction of states with u < 0")
ax.set_title("Task 2: Forecasting vs Switching OFF")
ax.grid(True, alpha=0.3)

# Task 3
ax = axes[1, 0]
sr_vals = [r[0] for r in task3_results]
gap_vals = [r[3] for r in task3_results]
ax.plot(sr_vals, gap_vals, 'o-', color='seagreen')
ax.set_xlabel("stddev_ratio (demand variance)")
ax.set_ylabel("V(Z_max=0) - V(Z_max=10)")
ax.set_title("Task 3: Deferral Benefit vs Variance")
ax.grid(True, alpha=0.3)

# Task 4
ax = axes[1, 1]
t3_vals = [r[0] for r in task4_results]
gap4_vals = [r[1] for r in task4_results]
ax.plot(t3_vals, gap4_vals, 'o-', color='mediumpurple')
ax.set_xlabel("theta3")
ax.set_ylabel("Avg idle servers (servers_on - v)")
ax.set_title("Task 4: theta3 vs Under-utilization")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_plots.png", dpi=150, bbox_inches='tight')
print("\nPlots saved to analysis_plots.png")
plt.show()