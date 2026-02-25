from utils import *
import policy.gaussian as gaus
import dataset.dataset as data
import experiments_single.imp_reward as ir


def train_kto(beta, delta = 1.5, good_ratio = None):

    policy = gaus.GaussianPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    y, labels = data.build_kto_dataset(delta, good_ratio = good_ratio)

    sigmas = []

    for _ in range(STEPS):
        optimizer.zero_grad()

        h = ir.implicit_reward(policy, y, beta)
        kl = policy.kl_to_ref().detach()  # no gradient through KL
        z = h - kl

        good_mask = labels == 1.0
        bad_mask = labels == 0.0

        v = torch.zeros_like(z)

        v[good_mask] = torch.sigmoid(z[good_mask])
        v[bad_mask] = torch.sigmoid(-LAMBDA * z[bad_mask])

        loss = torch.mean(1 - v)

        loss.backward()
        optimizer.step()

        sigmas.append(policy.sigma().item())

    return policy, sigmas