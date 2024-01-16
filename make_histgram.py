import matplotlib.pyplot as plt

# データセットの作成（例としてランダムな数値を使用）
data = [25.918539791924243,
25.92164901073097,
25.917213849576267,
25.923333941019084,
23.10012685862952,
25.934751606708428,
23.100127438115305,
25.93360590481304,
25.933014375614704,
23.1000872349574]

plt.hist(data, bins=10, edgecolor='black')

# グラフのタイトルと軸ラベルの設定（fontsizeパラメータを使用してサイズを指定）
plt.title("IMAGE72 random PSNR", fontsize=18)
plt.xlabel("PSNR", fontsize=18)
plt.ylabel("Frequency", fontsize=18)

# 目盛りのサイズを指定
plt.tick_params(axis='both', which='major', labelsize=17)

# グラフの表示
plt.show()