import json
import cv2
import os
import numpy as np
import math
import glob

num_of_human = 0
num_of_no = 0

for img in glob.glob("/home/kenchang/pyprac/FN/victor_11-02_test5/*.png"):
    cv_img = cv2.imread(img)
    #print(len(img))
    cv2.imshow("img", cv_img)
    while True:
            if cv2.waitKey(1) & 0xFF == ord('o'):
                num_of_human += 1
                print('num_of_human', num_of_human)
                break  

            if cv2.waitKey(1) & 0xFF == ord('x'):
                num_of_no += 1
                print('num_of_nohuman', num_of_no)
                break  

human_rate = num_of_human / (num_of_human + num_of_no)
noman_rate = num_of_no / (num_of_human + num_of_no)
print("human_rate=", human_rate, "noman_rate=", noman_rate)
