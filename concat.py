import cv2 
import os 
import glob
import numpy as np

fps = 5
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_name = 'optical.mp4'
path = '/home/kenchang/optical-flow_model/FastFlowNet/data/demo/'
ori_img_path = os.path.join(path, '2021-09-08_rgb-flow_floor-filtered/FN', '*.png')
bin_img_path = os.path.join(path, '2021-09-08_binary-flow_floor-filtered_thr1/FN', '*.png')
color_img_path = os.path.join(path, 'video4_flowtocolor', '*.png')
con_img_path = os.path.join(path, 'user15')
video_path = os.path.join(path, 'videos')

videoWriter = cv2.VideoWriter(video_path + '{}'.format(video_name),fourcc,fps,(800,600))
ori_imgs = sorted(glob.glob(ori_img_path))
bin_imgs = sorted(glob.glob(bin_img_path))
col_imgs = sorted(glob.glob(color_img_path))


for i in range(21):
    ori_img = cv2.imread(ori_imgs[i])
    bin_img = cv2.imread(bin_imgs[i])
    #concat = np.concatenate([bin_img, col_imgs], axis=1)
    concat = np.concatenate((ori_img, bin_img))
    cv2.imwrite(con_img_path + '/' + str(i) + ".png", concat)
    
    #videoWriter.write(concat )
#videoWriter.release()


