import cv2
import os
import glob
import datetime
 
# https://blog.csdn.net/m0_37733057/article/details/79023693
fps = 6

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#root_path = '/home/kenchang/anaconda3/envs/care-plus/care-plus-model-pipeline-activesample_dev/active_sample_output'
root_path = '/home/kenchang/anaconda3/envs/care-plus/care-plus-model-pipeline-activesample_dev'
username = 'user08'

#date = datetime.datetime.now()
date = '2021-01-31'
#img_path = os.path.join(root_path, 'active_sample_output', username, date, 'images', 'FN', '*.png')
#video_path = os.path.join(root_path, 'user_video')

""" #create folder
desired_folder = [os.path.join(video_path, 'user01'), os.path.join(video_path, 'user02'), os.path.join(video_path, 'user03'), 
                        os.path.join(video_path, 'user04'), os.path.join(video_path, 'user05'), os.path.join(video_path, 'user06'), os.path.join(video_path, 'user07'),
                         os.path.join(video_path, 'user08'), os.path.join(video_path, 'user09')]

for folder in desired_folder:
    if os.path.isdir(folder):
        print(folder + ' exist')
    else:
        os.makedirs(folder)
"""
"""
video_name = username + '_' + date + '.mp4'
video_path = os.path.join(video_path, username + '/')
print(video_path, type(video_path))


videoWriter = cv2.VideoWriter(video_path + '{}'.format(video_name),fourcc,fps,(800,600))
imgs = sorted(glob.glob(img_path))
"""

video_name = 'opfl.mp4'
#img_path = '~/optical-flow_model/FastFlowNet/data/frames'
#path = '/home/kenchang/optical-flow_model/FastFlowNet/data/frames_things3d_down'
path = '/home/kenchang//anaconda3/envs/care-plus/care-plus-model-pipeline-activesample_dev/active_sample/frames'
img_path = os.path.join(path, '*.png')
video_path = path + '/'
print(video_path, type(video_path))


videoWriter = cv2.VideoWriter(video_path + '{}'.format(video_name),fourcc,fps,(800,600))
imgs = sorted(glob.glob(img_path))

if len(imgs) > 0:
    print('start')
    for imgname in imgs:
        frame = cv2.imread(imgname)
        videoWriter.write(frame)
    videoWriter.release()