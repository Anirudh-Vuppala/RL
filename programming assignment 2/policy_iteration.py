import numpy as np
import time
from Assignment2Tools import prob_vector_generator

#HELPER FNS (MDP FORMULATION TO CODE)
def get_PERP(D):
    #to rep no forecast
    return D + 1


def get_num_states(S_max, Z_max, D, tau):
    #total no of states
    return (S_max + 1) * (Z_max + 1) * (D + 2) ** tau


def state_to_index(s, z, f_list, S_max, Z_max, D, tau):
    s = int(s)
    z = int(z)

    idx = s + (S_max + 1) * z
    stride = (S_max + 1) * (Z_max + 1)

    for k in range(tau):
        fk = int(f_list[k]) 
        idx += stride * fk
        stride *= (D + 2)

    return idx


def index_to_state(idx, S_max, Z_max, D, tau):
    # turn a single int to state format
    s = idx % (S_max + 1)
    idx //= (S_max + 1)

    z = idx % (Z_max + 1)
    idx //= (Z_max + 1)

    f_list = []
    for k in range(tau):
        f_list.append(idx % (D + 2))
        idx //= (D + 2)

    return s, z, f_list


def get_current_demand_dist(f_list, phi, D):
    #return (demand, prob)
    PERP = D + 1
    if tau_from_flist(f_list) == 0 or f_list[0] == PERP:
        return [(d, phi[d]) for d in range(D + 1)]
    else:
        return [(f_list[0], 1.0)]


def tau_from_flist(f_list):
    return len(f_list)


def get_action_space(s, z, d, S_max, Z_max):
    #r eturns list of valid (u, v) pairs for state (s,z) and seen demand d.
    actions = []
    u_min = -s                    # can't switch off more than are on
    u_max = S_max - s             # can't exceed S_max

    for u in range(u_min, u_max + 1):
        servers_available = s + u
        effective_demand = z + d

        v_min = max(0, effective_demand - Z_max)   # keep backlog <= Z_max
        v_max = min(servers_available, effective_demand)

        if v_min > v_max:
            continue

        for v in range(v_min, v_max + 1):
            actions.append((u, v))

    return actions


def immediate_cost(s, z, u, v, theta, alpha):
    #getting the immediate cost
    theta1, theta2, theta3 = theta
    return (theta1 * max(u, 0) + theta2 * (s + u) + theta3 * (v ** 2) + alpha * z)


def get_next_state_distribution(s, z, f_list, d, u, v, S_max, Z_max, D, tau, phi, lmbda):
    #to get a list of next state index, and probs for state transitions
    PERP = D + 1
    s_next = s + u
    z_next = z + d - v

    # Shift forecast vector: f'_k = f_{k+1} for k=1,...,tau-1
    # f_list[0] = f_1, f_list[1] = f_2, ..., f_list[tau-1] = f_tau
    # After shift: new f_list[0] = old f_list[1], ..., new f_list[tau-2] = old f_list[tau-1]
    # new f_list[tau-1] = f'_tau (random)

    if tau == 0:
        # No forecasts at all
        next_idx = state_to_index(s_next, z_next, [], S_max, Z_max, D, tau)
        return [(next_idx, 1.0)]

    # Shift:
    f_shifted = f_list[1:]  # length tau-1: [f_2, f_3, ..., f_tau]

    #random new forecast at tau
    lmbda_tau = lmbda[tau - 1]
    results = []

    #case 1: the forecast happens
    for d_prime in range(D + 1):
        prob = lmbda_tau * phi[d_prime]
        f_next = f_shifted + [d_prime]   # length tau
        next_idx = state_to_index(s_next, z_next, f_next, S_max, Z_max, D, tau)
        results.append((next_idx, prob))

    #case 2: le no forecast
    prob_perp = 1.0 - lmbda_tau
    f_next_perp = f_shifted + [PERP]     # length tau
    next_idx_perp = state_to_index(s_next, z_next, f_next_perp, S_max, Z_max, D, tau)
    results.append((next_idx_perp, prob_perp))

    return results

def policy_iteration(D, S_max, Z_max, tau, phi, lmbda, theta, alpha, beta, threshold, Kmin):


    num_states = get_num_states(S_max, Z_max, D, tau)

    #step 1 - init V to 0s, and init policy to first valid a
    V = np.zeros(num_states)
    policy = np.zeros((num_states, D + 1, 2), dtype=np.int32)

    for idx in range(num_states):
        s, z, f_list = index_to_state(idx, S_max, Z_max, D, tau)
        demand_dist = get_current_demand_dist(f_list, phi, D)
        for d, _ in demand_dist:
            actions = get_action_space(s, z, d, S_max, Z_max)
            policy[idx, d] = actions[0] if actions else (0, 0)

    


    converged = False
    iteration = 0

    total_start = time.time()



    # step 2:
    while not converged:
        iteration += 1
        iter_start = time.time()

        # step3 -  IPE (at least Kmin iters), then a conv check
        def run_one_eval_sweep(V_in):
            V_out = np.zeros_like(V_in)
            for idx in range(num_states):
                s, z, f_list = index_to_state(idx, S_max, Z_max, D, tau)
                demand_dist = get_current_demand_dist(f_list, phi, D)
                value = 0.0
                for d, prob_d in demand_dist:
                    u, v = policy[idx, d]
                    cost = immediate_cost(s, z, u, v, theta, alpha)
                    transitions = get_next_state_distribution(
                        s, z, f_list, d, u, v,
                        S_max, Z_max, D, tau, phi, lmbda
                    )
                    exp_next = sum(p * V_in[nxt] for nxt, p in transitions)
                    value += prob_d * (cost + beta * exp_next)
                V_out[idx] = value
            return V_out

        
        for _ in range(Kmin):
            V = run_one_eval_sweep(V)

        #do till conv.
        while True:
            V_new = run_one_eval_sweep(V)
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < threshold:
                break

        #step (4-6), policy improvement th.
        policy_new = np.copy(policy)

        for idx in range(num_states):
            s, z, f_list = index_to_state(idx, S_max, Z_max, D, tau)
            demand_dist = get_current_demand_dist(f_list, phi, D)

            for d, _ in demand_dist:
                actions = get_action_space(s, z, d, S_max, Z_max)
                best_action = tuple(policy[idx, d])
                best_q = float('inf')

                for u, v in actions:
                    cost = immediate_cost(s, z, u, v, theta, alpha)
                    transitions = get_next_state_distribution(
                        s, z, f_list, d, u, v,
                        S_max, Z_max, D, tau, phi, lmbda
                    )
                    exp_next = sum(p * V[nxt] for nxt, p in transitions)
                    q_val = cost + beta * exp_next

                    if q_val < best_q:
                        best_q = q_val
                        best_action = (u, v)

                policy_new[idx, d] = best_action

        #if no change, it means conv, le done
        if np.array_equal(policy_new, policy):
            converged = True

        #update...
        policy = policy_new

        print(f"Iteration {iteration} | Eval: {time.time()-iter_start:.1f}s | "
              f"Total so far: {time.time()-total_start:.1f}s")

    print(f"\nConverged in {iteration} iterations | "
          f"Total time: {time.time()-total_start:.1f}s")

    return V, policy


#params
D = 5
S_max = 15
theta = [5, 1, 0.25]

mu_d = 2
stddev_ratio = 0.6
stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
phi = prob_vector_generator(D, mu_d, stddev_d)

Z_max = 10
alpha = 0.5

tau = 3
lmbda = [1, 0.75, 0.5]

beta = 0.95
threshold = 0.1
Kmin = 10

V_optimal_pi, policy_optimal_pi = policy_iteration(
    D, S_max, Z_max, tau, phi, lmbda, theta, alpha, beta, threshold, Kmin
)

PERP = get_PERP(D)


print("Verifocation — f=[PERP]*tau states")


print(f"\n here z=0, and s varies: ")
print(f"{'s':>4} | {'d=0':>10} {'d=1':>10} {'d=2':>10} {'d=3':>10} {'d=4':>10} {'d=5':>10}")
print("-" * 74)
for s_test in range(6):
    idx = state_to_index(s_test, 0, [PERP]*tau, S_max, Z_max, D, tau)
    row = f"{s_test:>4} | "
    for d_test in range(D + 1):
        u, v = policy_optimal_pi[idx, d_test]
        row += f"({u:+d},{v:d})".rjust(10)
    print(row)

print(f"\n her s =3, and z varies : ")
print(f"{'z':>4} | {'d=0':>10} {'d=1':>10} {'d=2':>10} {'d=3':>10} {'d=4':>10} {'d=5':>10}")
print("-" * 74)
for z_test in range(5):
    idx = state_to_index(3, z_test, [PERP]*tau, S_max, Z_max, D, tau)
    row = f"{z_test:>4} | "
    for d_test in range(D + 1):
        u, v = policy_optimal_pi[idx, d_test]
        row += f"({u:+d},{v:d})".rjust(10)
    print(row)

print(f"\n val of V (the cost), z=0, f=PERP : ")
for s_test in range(S_max + 1):
    idx = state_to_index(s_test, 0, [PERP]*tau, S_max, Z_max, D, tau)
    print(f"  s={s_test:>2}, z=0: V = {V_optimal_pi[idx]:.4f}")