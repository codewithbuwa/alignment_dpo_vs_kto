from utils import *

class GaussianPolicy(nn.Module):
    def __init__(self, mu_init=5.0, log_sigma_init=math.log(2.0)):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu_init))
        self.log_sigma = nn.Parameter(torch.tensor(log_sigma_init))

    def sigma(self):
        return torch.exp(self.log_sigma)

    def log_prob(self, y):
        sigma = self.sigma()
        return -0.5 * (((y - self.mu) / sigma) ** 2 + 2 * self.log_sigma + math.log(2 * math.pi))
 
    def sample(self, n):
        return self.mu + self.sigma() * torch.randn(n)
    
    def kl_to_ref(self):
        sigma = self.sigma()
        return torch.log(REF_SIGMA / sigma) + \
               (sigma**2 + (self.mu - REF_MU)**2) / (2 * REF_SIGMA**2) - 0.5