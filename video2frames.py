import os
import cv2
import glob

user = 'user726'
date = '2022-01-28'
#category = 'wchair'  #wheelChair, diningCar, chair
#fold_path = './' + user + '/' + date + '/' + category + '_videos'
#output_path = './' + user + '/' + date + '/' + category + '_frames/'
fold_path = './' + user + '/' + date + '/' + 'videos'
output_path = './' + user + '/' + date + '/' + 'frames/'


if os.path.isdir(output_path):
    pass
else:
    os.mkdir(output_path)

i = 0

for files in sorted(glob.glob(os.path.join(fold_path, "*.mp4"))):
    print(files)
    cap = cv2.VideoCapture(files)
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()                                     
        if not ret: break
        frame = cv2.resize(frame, (400,300))
        #cv2.imwrite(output_path + date + '_' + str(i) + '_' + str(index) + ".png", frame)
        cv2.imwrite(output_path + str(index) + ".png", frame)
        index += 1

    i += 1