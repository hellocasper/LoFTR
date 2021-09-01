import os
# os.chdir("..")
# os.chdir(os.pardir)
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("/home/pengguohao/CV/pytorch/LoFTR/weights/indoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# Load example images
# img0_pth = "/home/pengguohao/CV/pytorch/LoFTR/assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
# img1_pth = "/home/pengguohao/CV/pytorch/LoFTR/assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
img0_pth = "/home/pengguohao/CV/pytorch/LoFTR/assets/samples/00008.jpg"
img1_pth = "/home/pengguohao/CV/pytorch/LoFTR/assets/samples/00009.jpg"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255. #[1,1,480,640]
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy() #(num_matches,2)
    mkpts1 = batch['mkpts1_f'].cpu().numpy() #(num_matches,2)
    mconf = batch['mconf'].cpu().numpy() #(num_matches,)
    
# Draw
color = cm.jet(mconf) #(num_matches,4)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)), #num_matches
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text)
print('done')