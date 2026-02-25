from __init__ import *

dpo_policy, dpo_sigmas = dp.train_dpo(BETA)
kto_policy, kto_sigmas = kt.train_kto(BETA)

plt.plot(dpo_sigmas, label="DPO")
plt.plot(kto_sigmas, label="KTO")
plt.xlabel("Training Step")
plt.ylabel("Sigma")
plt.legend()
plt.title("Entropy Dynamics")
plt.savefig("images/entropy_dynamics.png")
plt.show()
