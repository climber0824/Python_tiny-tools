import os
import random
import shutil
import argparse

def is_inp(name):
    return(name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])

def copyFile(fileDir, tarDir, select_num=130):
	pathDir = os.listdir(fileDir)
	sample = random.sample(pathDir, select_num)
	print(sample)
	for name in sample:
	   shutil.copyfile(fileDir+name, tarDir+name)

def random_select(user, date):
    fileDir = os.path.join(os.path.abspath(os.getcwd()) + '/' + user + '/' + date + '/FN_night/')
    outputDir = os.path.join(os.path.abspath(os.getcwd()))
    userDir = os.path.join(outputDir + '/' + user)
    tarDir = os.path.join(userDir + '/' + date + '_select' + '/')
    create_dir = True
    select_num = 150
    if create_dir == True:
        if os.path.isdir(outputDir):
            pass
        else:
            os.mkdir(outputDir)
        if os.path.isdir(userDir):
            pass
        else:
            os.mkdir(userDir)
        if os.path.isdir(tarDir):
            pass
        else:
            os.mkdir(tarDir)
    
    copyFile(fileDir, tarDir, select_num)
    print('random done', user, date)