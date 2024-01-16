import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.size"] = 20


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

# 点数のタプル
points = (p3, p4, p5, p6, p7, p8, p9, p10, p11)
"""
# 箱ひげ図
fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11"])

plt.title('PSNR')
plt.grid() # 横線ラインを入れることができます。

# 描画
plt.savefig(path + "hakohige_p.png")
plt.show()

points = (s3, s4, s5, s6, s7, s8, s9, s10, s11)
fig, ax = plt.subplots()
bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["ps", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"])

plt.title('SSIM')
plt.grid() # 横線ラインを入れることができます。

# 描画
plt.savefig(path + "hakohige_s.png")
plt.show()
"""

path = ("/mnt/data1/home/tatsumi/project_tatsumi/20231031/")
re_p = np.load(path + "data/re_p.npy")
re_s = np.load(path + "data/re_s.npy")

p30 = re_p[:, 0]
p50 = re_p[:, 1]
p80 = re_p[:, 2]
p120 = re_p[:, 3]
p150 = re_p[:, 4]
points = (p30, p50, p80, p120, p150)

fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["p30", "p50", "p80", "p120", "p150"])

plt.title('PSNR')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig(path + "hakohige_p.png")
plt.show()


s30 = re_s[:, 0]
s50 = re_s[:, 1]
s80 = re_s[:, 2]
s120 = re_s[:, 3]
s150 = re_s[:, 4]
points = (s30, s50, s80, s120, s150)

fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["s30", "s50", "s80", "s120", "s150"])

plt.title('SSIM')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig(path + "hakohige_s.png")
plt.show()

points = (p5, p6, p7, p8, p9,  p80)
fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["p5", "p6", "p7", "p8", "p9", "p80"])

plt.title('PSNR')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/hakohige_p.png")
plt.show()

points = (s5, s6, s7, s8, s9,  s80)
fig, ax = plt.subplots()

bp = ax.boxplot(points) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["s5", "s6", "s7", "s8", "s9", "s80"])

plt.title('SSIM')
plt.grid() # 横線ラインを入れることができます。

# 描画
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/hakohige_s.png")
plt.show()

path_1118 = "/mnt/data1/home/tatsumi/project_tatsumi/20231118/"
re_p = np.load(path_1118 + "re_p.npy")
re_s = np.load(path_1118 + "re_s.npy")
last_p = np.load(path_1118 + "last_p.npy")
last_s = np.load(path_1118 + "last_s.npy")
points = (re_s, last_s)
fig, ax = plt.subplots()
bp = ax.boxplot(points, showfliers=True) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["LRQMC", "DLRQMC"])
plt.title('SSIM')
plt.grid() # 横線ラインを入れることができます。
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/pre_last_s.png")
plt.show()

points = (re_p, last_p)
fig, ax = plt.subplots()
bp = ax.boxplot(points, showfliers = True) # 複数指定する場合はタプル型で渡します。
ax.set_xticklabels(["LRQMC", "DLRQMC"])
plt.title('PSNR')
plt.grid() # 横線ラインを入れることができます。
#plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/20231118/pre_last_p.png")
plt.show()

print(np.mean(s5))
print(np.mean(s6))
print(np.mean(s7))
print(np.mean(s8))
print(np.mean(s9))
print(np.mean(s80))