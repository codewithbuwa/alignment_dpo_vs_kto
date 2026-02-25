from utils import *
import policy.gaussian as gaus
import dataset.dataset as data
import experiments_single.imp_reward as ir


def train_kto(beta, delta=1.5, good_ratio=None, estimation_mode="analytical", alpha=0.5):
    policy = gaus.GaussianPolicy().to(DEVICE)
    ref_policy = gaus.GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    y, labels = data.build_kto_dataset(delta, good_ratio=good_ratio)

    sigmas = []
    # Initialize running KL at 0 or the analytical starting KL
    running_kl = torch.tensor(0.0).to(DEVICE) 

    for _ in range(STEPS):
        optimizer.zero_grad()

        log_p = policy.log_prob(y)
        log_p_ref = ref_policy.log_prob(y)
        h = ir.implicit_reward(policy, y, beta)

        # Operationalizing ref_KL
        if estimation_mode == "analytical":
            kl = policy.kl_to_ref().detach()
        elif estimation_mode == "batch":
            kl = torch.mean(log_p - log_p_ref).detach()
        elif estimation_mode == "running_avg":
            batch_kl = torch.mean(log_p - log_p_ref).detach()
            # Smooth update
            running_kl = (1 - alpha) * running_kl + alpha * batch_kl
            kl = running_kl
        elif estimation_mode == "fixed":
            kl = 0.1
        
        z = h - kl

        v = torch.zeros_like(z)
        v[labels == 1.0] = torch.sigmoid(z[labels == 1.0])
        v[labels == 0.0] = torch.sigmoid(-LAMBDA * z[labels == 0.0])

        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()

        sigmas.append(policy.sigma().item())

    return policy, sigmas