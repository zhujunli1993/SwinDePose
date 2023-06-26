import numpy as np
import pickle
import pdb
pdb.set_trace()
pkl_pth = '/workspace/DATA/OCCLUSION_LINEMOD/anns/ape/train.pkl'
with open(pkl_pth, 'rb') as f:
    data = pickle.load(f)
pkl_pth = '/workspace/DATA/OCCLUSION_LINEMOD/anns/ape/test.pkl'
with open(pkl_pth, 'rb') as f:
    data = pickle.load(f)