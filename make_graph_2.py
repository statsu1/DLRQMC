import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np

dr='3'

sns.set()
type = 'psnr'

pre_result = np.load('/mnt/data1/home/tatsumi/project_tatsumi/20230712/data/pre_psnr.npy')
last_result = np.load('/mnt/data1/home/tatsumi/project_tatsumi/20230712/data/last_psnr.npy')

fig = plt.figure()     # まずは図の大枠を作成
x = np.zeros(25)
y1 = np.zeros(25)
y2 = np.zeros(25)
#label_x = np.zeros(100, dtype = str)
for i in range(0,25):
  x[i] = i
for i in range(0, 25):
  y1[i] = last_result[i]
  y2[i] = pre_result[i]

plt.xlabel("image number", fontsize = 50)
plt.ylabel("PSNR", fontsize=50)
plt.tick_params(labelsize=40)

data = [y1, y2]
fig.set_size_inches(30, 10)
# マージンを設定
margin = 0.2  #0 <margin< 1
totoal_width = 1 - margin
 
# 棒グラフをプロット
for i, h in enumerate(data):
  pos = x - totoal_width *( 1- (2*i+1)/len(data) )/2
  plt.bar(pos, h, width = totoal_width/len(data))
 
# ラベルの設定
plt.xticks(x)
#plt.savefig('/mnt/data1/home/tatsumi/project_tatsumi/20230712/graph/psnr.pdf')
#plt.savefig('/mnt/data1/home/tatsumi/project_tatsumi/20230712/graph/psnr.png')





