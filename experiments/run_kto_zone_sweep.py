from __init__ import *

def run_delta_sweep(deltas):
    results = []

    for delta in deltas:

        policy, sigmas = kt.train_kto(BETA, delta)
        final_sigma = policy.sigma().item()

        results.append((delta, final_sigma))
        print(delta, final_sigma)

    deltas, sigmas = zip(*results)

    plt.figure()
    plt.plot(deltas, sigmas, marker='o')
    plt.xlabel("Delta (Zone Half-Width)")
    plt.ylabel("Final Sigma")
    plt.title("KTO: Final Sigma vs Delta")
    plt.savefig("images/kto_Final_Sigma_vs_Delta.png")
    plt.show()

    return results
delta_values = [0.2, 0.5, 1.0, 1.5, 2.0]
delta_results = run_delta_sweep(delta_values)