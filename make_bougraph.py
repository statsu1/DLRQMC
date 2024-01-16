import matplotlib.pyplot as plt
import numpy as np

# データ
x = ['unirand', 'grand', '1010', 'mean1', 'mean2', 'mean3', 'uni1', 'uni2', 'uni3']

shoki_psnr = np.load("/mnt/data1/home/tatsumi/project_tatsumi/shoki_result/shoki_psnr.npy")
y = shoki_psnr[0][0:9]

# 棒グラフの作成

plt.figure(figsize=(8, 4))
plt.bar(x, y)

# グラフのタイトルと軸ラベル
plt.title('IMAGE 0 PSNR', fontsize = 18)
plt.ylabel('PSNR', fontsize = 18)
plt.xlabel('initial value', fontsize = 18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.xticks(rotation=60, ha='center')

plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/shoki_result/shoki_psnr.png")

# グラフの表示
plt.show()

x = ['unirand', 'grand', '1010', 'mean1', 'mean2', 'mean3', 'uni1', 'uni2', 'uni3']

shoki_ssim = np.load("/mnt/data1/home/tatsumi/project_tatsumi/shoki_result/shoki_ssim.npy")
y = shoki_ssim[0][0:9]

# 棒グラフの作成
plt.figure(figsize=(8, 4))
plt.bar(x, y)

# グラフのタイトルと軸ラベル
plt.title('IMAGE 0 SSIM', fontsize = 18)
plt.ylabel('SSIM', fontsize = 18)
plt.xlabel('initial value', fontsize = 18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.xticks(rotation=60, ha='center')

plt.savefig("/mnt/data1/home/tatsumi/project_tatsumi/shoki_result/shoki_ssim.png")

# グラフの表示
plt.show()