
import torch
from backends.dla import build
from backends.decode import decode_detection
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

device = torch.device('cuda')

backend = build(num_classes=6,num_layers=34)
# checkpoint = torch.load('/home/nides/Projects/centernet-uda/outputs/default/model_best.pth')
# backend.load_state_dict(checkpoint['state_dict'])
_ = backend.eval()
backend.cuda()
x = torch.randn((1, 3, 512, 512), requires_grad=True).cuda()
out = backend(x)

print(out)

