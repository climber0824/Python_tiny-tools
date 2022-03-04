import os
import glob
import re
import cv2

path = './labels/'
files = sorted(os.listdir(path))
img = cv2.imread('./8_24.png')


p1 = re.compile(r'[(](.*?)[)]', re.S)
cX = []
cY = []
total_x = []
total_y = []
color = (0, 0, 255)
radius = 2

for txt in files:
    position = path + txt
    with open(position, 'r') as f:
        for line in f.readlines():
            #print(line)
            x = re.findall(p1, line)
            cX = []
            cY = []
            for i in range(len(x)):
                cX.append(re.findall(p1, line)[i].split(',')[0])
                cY.append(re.findall(p1, line)[i].split(',')[1])
            
            if len(x) == 2:
                centerX = (int(cX[0]) + int(cX[1])) / 2
                centerY = (int(cY[0]) + int(cY[1])) / 2
                #print('center', centerX, centerY)
                total_x.append(centerX)
                total_y.append(centerY)
            if len(x) == 4:
                centerX1 = (int(cX[0]) + int(cX[1])) / 2
                centerX2 = (int(cX[2]) + int(cX[3])) / 2
                centerY1 = (int(cY[0]) + int(cY[1])) / 2
                centerY2 = (int(cY[2]) + int(cY[3])) / 2
                #print('center2', centerX1, centerY1, centerX2, centerY2)
                total_x.append(centerX1)
                total_y.append(centerY1)
                total_x.append(centerX2)
                total_y.append(centerY2)
            
for i in range(len(total_x)):
    #print(total_x[i], total_y[i])
    img = cv2.circle(img, (int(total_x[i]), int(total_y[i])), radius, color, -1)

cv2.imwrite('heat_map.png', img)
cv2.imshow("img", img)
cv2.waitKey(0)