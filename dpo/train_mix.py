from policy.gaussian_mixture import GaussianMixturePolicy
from utils import *

y_w, y_l = data.build_dpo_dataset()
REF_POLICY = GaussianMixturePolicy().REF_POLICY

def train_dpo_mixture(ref_policy: GaussianMixturePolicy = REF_POLICY, 
                      beta = BETA, n_components=2, steps=STEPS, lr=LR):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    sigmas = []  # track average sigma for monitoring

    for _ in range(steps):
        optimizer.zero_grad()
        log_pi_w = policy.log_prob(y_w)
        log_pi_l = policy.log_prob(y_l)
        log_ref_w = ref_policy.log_prob(y_w)
        log_ref_l = ref_policy.log_prob(y_l)

        h_w = beta * (log_pi_w - log_ref_w)
        h_l = beta * (log_pi_l - log_ref_l)

        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_sigma = policy.sigmas().mean().item()
            sigmas.append(avg_sigma)

    return policy, sigmas
