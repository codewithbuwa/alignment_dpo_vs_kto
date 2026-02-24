from __init__ import *
from policy.gaussian import *
from utils import *
import math

def plot_density_overlay(dpo_policy : GaussianPolicy, kto_policy: GaussianPolicy):
    y_vals = torch.linspace(-2, 14, 1000)

    def gaussian_pdf(mu, sigma, y):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * \
               torch.exp(-0.5 * ((y - mu) / sigma) ** 2)

    ref_pdf = gaussian_pdf(REF_MU, REF_SIGMA, y_vals)
    dpo_pdf = gaussian_pdf(dpo_policy.mu.item(),
                           dpo_policy.sigma().item(),
                           y_vals)
    kto_pdf = gaussian_pdf(kto_policy.mu.item(),
                           kto_policy.sigma().item(),
                           y_vals)

    plt.figure()
    plt.plot(y_vals, ref_pdf, label="Reference")
    plt.plot(y_vals, dpo_pdf, label="DPO")
    plt.plot(y_vals, kto_pdf, label="KTO")
    plt.legend()
    plt.title("Density Projection")
    plt.savefig("images/density_projection.png")
    plt.show()

dpo_policy = dp.train_dpo(beta=1.0)[0]
kto_policy = kt.train_kto(beta=1.0)[0]
plot_density_overlay(dpo_policy, kto_policy)