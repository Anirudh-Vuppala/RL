import numpy as np
from Assignment2Tools import prob_vector_generator
from policy_iteration import policy_iteration

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




# params (same as policy_iteration)
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




# type 1 → expected switching cost
def policy_evaluation_type1(policy):
    num_states = get_num_states(S_max, Z_max, D, tau)
    V = np.zeros(num_states)
    x_ref = 0

    def sweep(V_in):
        V_out = np.zeros_like(V_in)

        for idx in range(num_states):
            s, z, f_list = index_to_state(idx, S_max, Z_max, D, tau)
            demand_dist = get_current_demand_dist(f_list, phi, D)

            val = 0.0

            for d, prob_d in demand_dist:
                u, v = policy[idx, d]

                # switching cost only when u > 0
                r = theta[0] * max(u, 0)

                trans = get_next_state_distribution(
                    s, z, f_list, d, u, v,
                    S_max, Z_max, D, tau, phi, lmbda
                )

                exp_next = sum(p * V_in[nxt] for nxt, p in trans)
                val += prob_d * (r + exp_next)

            V_out[idx] = val

        return V_out

    for _ in range(Kmin):
        V = sweep(V)

    while True:
        V_new = sweep(V)

        mu = V_new[x_ref]
        V_new -= mu

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < threshold:
            break

    return mu



# type 2 → P(servers > threshold)
def policy_evaluation_type2(policy, threshold_s=5):
    num_states = get_num_states(S_max, Z_max, D, tau)
    V = np.zeros(num_states)
    x_ref = 0

    def sweep(V_in):
        V_out = np.zeros_like(V_in)

        for idx in range(num_states):
            s, z, f_list = index_to_state(idx, S_max, Z_max, D, tau)
            demand_dist = get_current_demand_dist(f_list, phi, D)

            val = 0.0

            for d, prob_d in demand_dist:
                u, v = policy[idx, d]

                # indicator → 1 if condition true else 0
                r = 1 if s > threshold_s else 0

                trans = get_next_state_distribution(
                    s, z, f_list, d, u, v,
                    S_max, Z_max, D, tau, phi, lmbda
                )

                exp_next = sum(p * V_in[nxt] for nxt, p in trans)
                val += prob_d * (r + exp_next)

            V_out[idx] = val

        return V_out

    for _ in range(Kmin):
        V = sweep(V)

    while True:
        V_new = sweep(V)

        mu = V_new[x_ref]
        V_new -= mu

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < threshold:
            break

    return mu



try:
    policy = policy_optimal_pi

    print("\n--- policy evaluation ---")

    res1 = policy_evaluation_type1(policy)
    print("expected switching cost:", round(res1, 4))

    res2 = policy_evaluation_type2(policy, threshold_s=5)
    print("P(servers > 5):", round(res2, 4))

except NameError:
    print("run policy_iteration first to get policy_optimal_pi")