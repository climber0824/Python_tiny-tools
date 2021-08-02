import os
import numpy
import cv2
import random
import shutil

def is_inp(name):
    return(name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])


def copyFile(fileDir, tarDir):
    # 1
	pathDir = os.listdir(fileDir)

    # 2
	sample = random.sample(pathDir, 30)
	print(sample)
	
	# 3
	for name in sample:
	   shutil.copyfile(fileDir+name, tarDir+name)


if __name__ == "__main__":
    user = 'user726'
    fileDir = './user_pic/' + user + '/'
    date = '06-13/'
    randomDir = './user_pic/' + user + '/random_select/'
    tarDir = './user_pic/' + user + '/random_select/'
    create_dir = True

    copyFile(fileDir + date, tarDir + date)
    """
    if create_dir == True:
        for path in os.listdir(fileDir + 'self_select'):
            print((path))
            os.mkdir(randomDir+path)
    
    for path in os.listdir(fileDir + 'self_select'):
        copyFile(fileDir + path + '/', tarDir + path + '/')
    """
        