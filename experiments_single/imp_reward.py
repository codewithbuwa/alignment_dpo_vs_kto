import math
from utils import *
from policy.gaussian import *
from policy.gaussian_mixture import *

def implicit_reward(policy, ref_policy, y, beta):

    log_pi = policy.log_prob(y)
    log_ref = ref_policy.log_prob(y)
    
    return beta * (log_pi - log_ref)