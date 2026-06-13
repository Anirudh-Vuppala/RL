import numpy as np

try:
    # Load the saved reward arrays
    # (Update the filenames if yours are slightly different!)
    r1 = np.load('rewards1.npy')
    r2 = np.load('rewards2.npy')

    print("==========================================")
    print("         TESTING REWARD AVERAGES          ")
    print("==========================================")
    print(f"Total episodes tracked: {len(r1)}")

    print("\n--- Policy 1 (Standard Q-Learning) ---")
    print(f"Overall Average (All episodes):        {r1.mean():.4f}")
    print(f"Converged Average (Last 50 episodes):  {r1[-50:].mean():.4f} (± {r1[-50:].std():.4f})")

    print("\n--- Policy 2 (Structural Knowledge) ---")
    print(f"Overall Average (All episodes):        {r2.mean():.4f}")
    print(f"Converged Average (Last 50 episodes):  {r2[-50:].mean():.4f} (± {r2[-50:].std():.4f})")
    
    print("\n==========================================")
    
    # Optional: Check if one strictly beats the other at the end
    diff = r2[-50:].mean() - r1[-50:].mean()
    if diff > 0:
        print(f"🏆 Policy 2 beats Policy 1 by {diff:.4f} on average at convergence.")
    elif diff < 0:
        print(f"🏆 Policy 1 beats Policy 2 by {abs(diff):.4f} on average at convergence.")
    else:
        print("🤝 Both policies converged to the exact same average reward.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Make sure 'rewards1.npy' and 'rewards2.npy' are in the same directory as this script.")