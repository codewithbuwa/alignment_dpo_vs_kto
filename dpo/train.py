from policy.gaussian import *
from experiments_single.imp_reward import implicit_reward
from utils import *
from dataset.dataset import *


yw, yl = build_dpo_dataset()
ref_policy = GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)

def train_dpo(beta, w = yw, l = yl):
    policy = GaussianPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    sigmas = []

    for _ in range(STEPS):
        optimizer.zero_grad()
        hw = implicit_reward(policy, ref_policy, yw, beta)
        hl = implicit_reward(policy, ref_policy, yl, beta)

        bt = torch.sigmoid(hw - hl)
        loss = -torch.mean(torch.log(bt))

        loss.backward()
        optimizer.step()

        sigmas.append(policy.sigma().item())

    return policy, sigmas