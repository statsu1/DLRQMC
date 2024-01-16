import matplotlib.pyplot as plt
import numpy as np


mean = 0.5
std = 0.1  # ガウス分布の標準偏差を調整して範囲を制御
size = 32100  # 生成する乱数の個数

gaussian_values = np.random.normal(mean, std, size)

# 範囲の制約
gaussian_values = np.clip(gaussian_values, 0, 1)

# 結果の表示


plt.hist(gaussian_values, bins=30)  # ヒストグラムのビン数を指定

# プロットの装飾
plt.title('Histogram')  # グラフのタイトル
plt.xlabel('Value')  # x軸のラベル
plt.ylabel('Frequency')  # y軸のラベル

# グラフの表示
plt.show()

U_gau = np.random.normal(mean, std, (642, 50))
V_gau = np.random.normal(mean, std, (50, 962))

np.save("/mnt/data1/home/tatsumi/project_tatsumi/shoki_mat/U_gau", U_gau)
np.save("/mnt/data1/home/tatsumi/project_tatsumi/shoki_mat/V_gau", V_gau)
