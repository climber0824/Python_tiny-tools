import os
import numpy
import cv2
def is_inp(name):
    return(name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])

date = '07-26'
user = 'user10'
#pic_path = '/home/kenchang/pyprac/user_pic/user15/05-05_select'
pic_path = '/home/kenchang/pyprac/user_pic/'+ user + '/' + date + '_select'
tarDir = '/home/kenchang/pyprac/user_pic/' + user + '/' + date + '/' 
#pic_path = '/home/kenchang/pyprac/user_pic/'+ user + '/' + '06-11_select'
#target_path = '/home/kenchang/pyprac/total_pic'
os.mkdir(tarDir)
all_inps = os.listdir(pic_path)
all_inp = [i for i in all_inps if is_inp(i)]
for i in range(len(all_inp)):
    path_=os.path.join(pic_path,all_inp[i])
    
    I=cv2.imread(path_)
    #cv2.imwrite('/home/kenchang/pyprac/user_pic/user15/05-05/%06d'%(i)+'.png',I)      #按照00000~以此排序
    cv2.imwrite('/home/kenchang/pyprac/user_pic/' + user + '/' + date + '/' + '%06d'%(i)+'.png',I)      #按照00000~以此排序
    #cv2.imwrite('/home/kenchang/pyprac/user_pic/' + user + '/' + '06-11' + '/' + '%06d'%(i)+'.png',I)      #按照00000~以此排序
    #cv2.imwrite('/home/kenchang/pyprac/total_pic/'+'{}'.format(i)+'.png',I)   #按照1~以此排序
    #cv2.imwrite('/home/kenchang/pyprac/total_pic/'+all_inp[i],I)                #按照原命名排序 