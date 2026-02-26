from dataset.dataset import *
from policy.gaussian import *
from utils import *
import experiments_single.imp_reward as ir

ref_policy = GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)
yw, yl = build_dpo_dataset()

def train_dpo(beta, w = yw, l = yl):
    policy = GaussianPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    sigmas = []

    for _ in range(STEPS):
        optimizer.zero_grad()
        hw = ir.implicit_reward(policy, ref_policy, yw, beta)
        hl = ir.implicit_reward(policy, ref_policy, yl, beta)

        bt = torch.sigmoid(hw - hl)
        loss = -torch.mean(torch.log(bt))

        loss.backward()
        optimizer.step()

        sigmas.append(policy.sigma().item())

    return policy, sigmas