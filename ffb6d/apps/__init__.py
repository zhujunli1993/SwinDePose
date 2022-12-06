import torch
import random
import numpy as np
# set random seed
# seed = 42
seed = 80
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)