import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np

dr='3'

path = "/mnt/data1/home/tatsumi/project_tatsumi/result/20240227142320"
name = path + "/result"
aa = np.load(name + ".npy")

sns.set()
type = 'psnr'

pre_result = aa[0]
last_result = aa[2]

for ii in [54, 62, 63]:
  print(pre_result[ii], last_result[ii])

fig = plt.figure()     # まずは図の大枠を作成
x1 = np.zeros(100)
y1 = np.zeros(100)
#label_x = np.zeros(100, dtype = str)
for i in range(0,100):
  x1[i] = i
for i in range(0, 100):
  y1[i] = last_result[i] - pre_result[i]
plt.ylim(-0.02,0.02)
plt.xlabel("image number", fontsize = 80)
plt.ylabel(r"$S_{\mathrm{PSNR}}$", fontsize=80)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks(np.arange(-0.25, 0.26, step=0.05))
plt.bar(x1, y1, color='b', width=1, align="center")
plt.tick_params(labelsize=60)
fig.patch.set_facecolor('white')
fig.set_size_inches(30, 10)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.21, top=0.9)
plt.savefig(path+'/psnr.pdf')
plt.savefig(path+'/psnr.png')

sns.set()
type = 'ssim'
pre_result = aa[1]
last_result = aa[3]

for ii in [54, 62, 63]:
  print(pre_result[ii], last_result[ii])

fig = plt.figure()     # まずは図の大枠を作成
x1 = np.zeros(100)
y1 = np.zeros(100)
#label_x = np.zeros(100, dtype = str)
for i in range(0,100):
  x1[i] = i
for i in range(0, 100):
  y1[i] = last_result[i] - pre_result[i]
plt.ylim(-0.02,0.02)
plt.xlabel("image number", fontsize = 80)
plt.ylabel(r"$S_{\mathrm{SSIM}}$", fontsize=80)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks(np.arange(-0.02, 0.021, step=0.005))
plt.bar(x1, y1, color='b', width=1, align="center")
plt.tick_params(labelsize=60)
fig.patch.set_facecolor('white')
fig.set_size_inches(30, 10)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.21, top=0.9)
plt.savefig(path+'/ssim.pdf')
plt.savefig(path+'/ssim.png')



