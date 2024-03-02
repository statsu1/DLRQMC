import cv2
import numpy as np
import sys
sys.path.append('/mnt/data1/home/tatsumi/project_tatsumi/BoostingMonocularDepth')  # 'lrqmc.py' ファイルが存在するディレクトリへのパスを指定
import lrqmc
import os
import subprocess
from skimage.metrics import structural_similarity as SSIM
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

#9, 19, 65
"""
path = "/mnt/data1/home/tatsumi/project_tatsumi/20231118/"

re_p = np.load(path  + "re_p.npy")
re_s = np.load(path  + "re_s.npy")

last_p = np.load(path  + "last_p.npy")
last_s = np.load(path  + "last_s.npy")
print(np.mean(re_p), np.mean(re_s))
print(np.mean(last_p), np.mean(last_s))
"""

path = "/mnt/data1/home/tatsumi/project_tatsumi/result/20240227142320"
aa = np.load(path + "/result.npy")

sa_p = aa[2]-aa[0]
sa_s = aa[3]-aa[1]

num_p = 0
num_s = 0

for i in range(0, 100):
    if(sa_p[i] >= 0):
        num_p += 1
    if(sa_s[i] > 0):
        num_s += 1

print(num_p, num_s)
#print(aa)