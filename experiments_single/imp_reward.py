import math
from utils import *
from policy.gaussian import *

def implicit_reward(policy: GaussianPolicy, y, beta):
    log_pi = policy.log_prob(y)
    log_ref = -0.5 * (((y - REF_MU) / REF_SIGMA) ** 2 +
                      math.log(2 * math.pi * REF_SIGMA**2))
    
    return beta * (log_pi - log_ref)