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



#commenttt

path = "/mnt/data1/home/tatsumi/project_tatsumi/images"
data_dir = "/mnt/data1/home/tatsumi/project_tatsumi/BoostingMonocularDepth/inputs"
output_dir = "/mnt/data1/home/tatsumi/project_tatsumi/BoostingMonocularDepth/outputs_leres"
shoki_types = ["med1", "med2","random_fixed","uni_fill1", "uni_fill2"]
result_dir = "/mnt/data1/home/tatsumi/project_tatsumi/result"

new_dir = lrqmc.generate_timestamp_filename('')
os.mkdir(new_dir)
print(new_dir)
os.mkdir(new_dir + "/images")


INIT_RANK = [30, 50, 80, 120, 150, 321]

dr = '3'
date = "20231031"

shoki_type = "random_fixed"
threshold = 3000
savepath = "/mnt/data1/home/tatsumi/project_tatsumi/" + date +"/"

zero = np.zeros(100)
lrqmc_p = np.zeros(100)
lrqmc_s = np.zeros(100)
dlrqmc_p = np.zeros(100)
dlrqmc_s = np.zeros(100)


"""
#1031の保存
#np.save("/mnt/data1/home/tatsumi/project_tatsumi/20231031/data/re_p.npy", zero)

#1031の読み込み
#last_p = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231031/data/last_p.npy")
#last_s = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231031/data/last_s.npy")
re_p = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231118/re_p.npy")
re_s = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231118/re_s.npy")
last_p = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231118/last_p.npy")
last_s = np.load("/mnt/data1/home/tatsumi/project_tatsumi/20231118/last_s.npy")
"""



#np.save('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_psnr.npy', zero)
#np.save('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_ssim.npy', zero)
#np.save('/mnt/data1/home/tatsumi/project_tatsumi/20230908/uni_10_p.npy', zero)
#np.save('/mnt/data1/home/tatsumi/project_tatsumi/20230908/uni_10_s.npy', zero)
#np.save(savepath + "_p.npy", zero)
#np.save(savepath + "_s.npy", zero)

#pre_psnr = np.load('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_psnr.npy')
#pre_ssim = np.load('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_ssim.npy')
#last_psnr = np.load(savepath + '_p.npy')
#last_ssim = np.load(savepath + '_s.npy')
#print(last_psnr[0])
#print(last_ssim[0])

#print(threshold)


sum = np.zeros(5)

for imgnum in range(0,100):
    img = cv2.imread(path + '/original/' + str(imgnum) + ".jpg")
    qimg = lrqmc.img2qm(path + '/original/' + str(imgnum) + ".jpg")
    img_masked, mask = lrqmc.add_random_missing_pixels(qimg, 0.1 * int(dr), "uniform", 1)
    
    masked_bgr = lrqmc.qm2img(img_masked)[:, :, ::-1] * 255
    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/masked/'+ str(imgnum) + "_masked.jpg", masked_bgr)
    cv2.imwrite(new_dir + "/images/" + str(imgnum) + "_masked.jpg", masked_bgr)
    i = 0

    X, U, V = lrqmc.lrqmc_2(img_masked, mask, 80, 2, 1e-3, 100, 1e-3, True, threshold, 0.9, False, False, "random_fixed")

    recovered_bgr = lrqmc.qm2img(X)[:, :, ::-1] * 255
    cv2.imwrite(new_dir + "/images/" + str(imgnum) + "_recovered.jpg", recovered_bgr)
    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/recovered/' + str(imgnum) + "_re_"+str(init_rank)+".jpg", recovered_bgr)
    p3 = cv2.PSNR(lrqmc.qm2img(qimg), lrqmc.qm2img(X))/3
    s3 = SSIM(lrqmc.qm2img(qimg), lrqmc.qm2img(X), data_range = 1, channel_axis = 2)
    print(imgnum, "{:.3f}".format(p3), "{:.3f}".format(s3))

    lrqmc_p[imgnum] = cv2.PSNR(lrqmc.qm2img(qimg), lrqmc.qm2img(X))/3
    lrqmc_s[imgnum] = SSIM(lrqmc.qm2img(qimg), lrqmc.qm2img(X), data_range = 1, channel_axis = 2)
    
    lrqmc.save_image(X, data_dir + "/depth.jpg")

    os.chdir('/mnt/data1/home/tatsumi/project_tatsumi/BoostingMonocularDepth')

    subprocess.run(['python3', 'run.py', '--Final', '--data_dir', data_dir, '--output_dir', output_dir, '--depthNet', '2'])
    img_uint8 = cv2.imread(output_dir + '/depth.png', cv2.CV_8U) # 白黒画像として読み込み

    depth = img_uint8.astype(np.float64) # floatに型変換
    depth = 255-depth
    cv2.imwrite(new_dir + "/images/" + str(imgnum) + "_depth.jpg", depth)
    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/depth/'+str(imgnum) + "_depth.jpg", depth)
    
    height = img.shape[0]
    width = img.shape[1]

    #二回目の復元の虚部を，一回目の復元結果にするバージョン
    if(1):
        X_d = np.copy(img_masked)
        for y in range(0, height):
            for x in range(0, width):
                if(X_d[y][x][1] != 0):
                    X_d[y][x][0] = depth[y][x]/255
                else:
                    X_d[y][x][0] = 0
        #print(X_d[0])
        #print(X_d.shape)
    else:
        print("深度を欠損させない")
        X_d = np.copy(img_masked)
        for y in range(0, height):
            for x in range(0, width):
                    X_d[y][x][0] = depth[y][x]/255
        
        #print(X_d[0])
        #print(X_d.shape)

    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/for_ppt.png', X_d[:,:,0]*255)



    X_d_re, U, V = lrqmc.lrqmc_2(X_d, mask, 80, 2, 1e-3, 100, 1e-3, True, threshold, 0.9, False, False, "random_fixed")
    
    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/last_re.png', lrqmc.qm2real(X_d_re)*255)

    last_bgr = lrqmc.qm2img(X_d_re)[:, :, ::-1] * 255
    cv2.imwrite(new_dir + "/images/" + str(imgnum) + "_last.jpg", last_bgr)
    #cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/last/' + str(imgnum) + "_last_"+str(init_rank)+".jpg", last_bgr)


    print("LRQMC後", cv2.PSNR(lrqmc.qm2img(qimg), lrqmc.qm2img(X))/3, SSIM(lrqmc.qm2img(qimg), lrqmc.qm2img(X), data_range = 1, channel_axis = 2))
    print("最終", cv2.PSNR(lrqmc.qm2img(qimg), lrqmc.qm2img(X_d_re))/3, SSIM(lrqmc.qm2img(qimg), lrqmc.qm2img(X_d_re), data_range = 1, channel_axis = 2))

    dlrqmc_p[imgnum] = cv2.PSNR(lrqmc.qm2img(qimg), lrqmc.qm2img(X_d_re))/3
    dlrqmc_s[imgnum] = SSIM(lrqmc.qm2img(qimg), lrqmc.qm2img(X_d_re), data_range = 1, channel_axis = 2)

cv2.imwrite("/mnt/data1/home/tatsumi/project_tatsumi/nofull_depth.jpg", lrqmc.qm2img(X_d_re)[:, :, ::-1] * 255)
#cv2.imwrite("/mnt/data1/home/tatsumi/project_tatsumi/full_depth_2.jpg",X_d_re*255)
result_all = np.stack([lrqmc_p, lrqmc_s, dlrqmc_p, dlrqmc_s])
np.save(new_dir + "/result.npy", result_all)

#np.save(filename, result_all)

print(result_all)




"""


#img_1 = cv2.imread('/mnt/data1/home/tatsumi/project_tatsumi/20230712/last/82_last.jpg',0)
#img_2 = cv2.imread('/mnt/data1/home/tatsumi/project_tatsumi/20230712/recovered/82_re.jpg',0)
#img_diff = cv2.absdiff(img_1, img_2)
#cv2.imwrite('/mnt/data1/home/tatsumi/project_tatsumi/20230712/sabun82.jpg',img_diff)




#np.save('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_psnr.npy', pre_psnr)
#np.save('/mnt/data1/home/tatsumi/project_tatsumi/' + date + '/data/pre_ssim.npy', pre_ssim)

"""









