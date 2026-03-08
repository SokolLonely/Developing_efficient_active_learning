import torch
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ROOT_DIR
pth = os.path.join(ROOT_DIR, 'data', 'ALDH1', 'screen')
x_ecfp = torch.load(os.path.join(pth, 'x'), weights_only=False)
print(f"ecfp len = {len(x_ecfp)}")
x_m = torch.load(os.path.join(pth, 'x_morfeus'), weights_only=False)#.squeeze(-1)
print(f"morfeus len = {len(x_m)}")
x_m = x_m.squeeze(-1)
print(f"squeezed morfeus len = {len(x_m)}")
x = np.concatenate((x_ecfp, x_m), axis = 1)
print(f"Y LEN = {len(y)}")