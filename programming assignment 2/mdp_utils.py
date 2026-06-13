import numpy as np


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



if __name__ == "__main__":
    #assignment params
    D = 2
    S_max = 3
    Z_max = 2
    tau = 0
    PERP = get_PERP(D)

    from Assignment2Tools import prob_vector_generator
    import numpy as np
    mu_d = 2
    stddev_ratio = 0.6
    stddev_d = stddev_ratio * np.sqrt(D * (D - mu_d))
    phi = prob_vector_generator(D, mu_d, stddev_d)
    lmbda = [1, 0.75, 0.5]
    theta = [5, 1, 0.25]
    alpha = 0.5

    num_states = get_num_states(S_max, Z_max, D, tau)
    print(f"Total states: {num_states}")
    # Expected: 16 * 11 * 7^3 = 16 * 11 * 343 = 60,368

    # Test encode/decode roundtrip
    s, z, f_list = 3, 2, []
    idx = state_to_index(s, z, f_list, S_max, Z_max, D, tau)
    s2, z2, f2 = index_to_state(idx, S_max, Z_max, D, tau)
    assert (s, z, f_list) == (s2, z2, f2), "encode decode doesnt match..."
    print(f"State ({s},{z},{f_list}) -> index {idx} -> ({s2},{z2},{f2}) ✓")

    # Test action space
    actions = get_action_space(s=3, z=2, d=4, S_max=S_max, Z_max=Z_max)
    print(f"Action space for (s=3,z=2,d=4): {len(actions)} actions")

    # Test immediate cost
    c = immediate_cost(s=3, z=2, u=2, v=3, theta=theta, alpha=alpha)
    print(f"Immediate cost (s=3,z=2,u=2,v=3): {c}")

    # Test transition distribution (probs should sum to 1)
    dist = get_next_state_distribution(
        s=3, z=2, f_list=[1, PERP, 4], d=4, u=2, v=3,
        S_max=S_max, Z_max=Z_max, D=D, tau=tau, phi=phi, lmbda=lmbda
    )
    total_prob = sum(p for _, p in dist)
    print(f"Transition prob sum: {total_prob:.6f}") #shd be 1
    print("doneee")