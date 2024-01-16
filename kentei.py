import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from scipy import stats
plt.rcParams["font.size"] = 25


#20231116の、最大固有値に対する割合で削減するバージョンの閾値どうするか
path = "/mnt/data1/home/tatsumi/project_tatsumi/20231116/"

data_p = np.load(path + "re_p.npy")
data_s = np.load(path + "re_s.npy")
p3 = data_p[:, 0]
p4 = data_p[:, 1]
p5 = data_p[:, 2]
p6 = data_p[:, 3]
p7 = data_p[:, 4]
p8 = data_p[:, 5]
p9 = data_p[:, 6]
p10 = data_p[:, 7]
p11 = data_p[:, 8]

s3 = data_s[:, 0]
s4 = data_s[:, 1]
s5 = data_s[:, 2]
s6 = data_s[:, 3]
s7 = data_s[:, 4]
s8 = data_s[:, 5]
s9 = data_s[:, 6]
s10 = data_s[:, 7]
s11 = data_s[:, 8]


path = ("/mnt/data1/home/tatsumi/project_tatsumi/20231031/")
re_p = np.load(path + "data/re_p.npy")
re_s = np.load(path + "data/re_s.npy")

p30 = re_p[:, 0]
p50 = re_p[:, 1]
p80 = re_p[:, 2]
p120 = re_p[:, 3]
p150 = re_p[:, 4]
points = (p30, p50, p80, p120, p150)

s30 = re_s[:, 0]
s50 = re_s[:, 1]
s80 = re_s[:, 2]
s120 = re_s[:, 3]
s150 = re_s[:, 4]
points = (s30, s50, s80, s120, s150)



points = (p5, p6, p7, p8, p9,  p80)
fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["p5", "p6", "p7", "p8", "p9", "p80"])

plt.title('PSNR')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/hakohige_p.png")
plt.show()

points = (s3, s6, s7, s8, s9,  s80)
fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["s3", "s6", "s7", "s8", "s9", "s80"])

plt.title('SSIM')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/hakohige_s.png")
plt.show()


import matplotlib.pyplot as plt

# データの準備
data = [s3, s6, s7, s8, s9, s80]
labels = ["s3", "s6", "s7", "s8", "s9", "s80"]

# ボンフェローニ補正などの多重比較のための有意水準
alpha = 0.05

# 箱ひげ図の描画
fig, ax = plt.subplots()
bp = ax.boxplot(data)
ax.set_xticklabels(labels)
plt.title('SSIM')
plt.grid()

# 有意水準以下の場合に'*'を表示
for i, combo in enumerate(combinations(range(len(data)), 2)):
    group1 = data[combo[0]]
    group2 = data[combo[1]]
    
    statistic, p_value = stats.ttest_ind(group1, group2)
    
    if p_value < alpha:
        x_pos = [i+1]
        y_pos = max(max(group1), max(group2))
        ax.text(x_pos, y_pos, '*', ha='center', va='bottom', color='red', fontsize=12)

plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/hakohige_s_2.png")
plt.show()

