import os
path = '../previous_datas_test/user02/2021-12-17'

fileList = os.listdir(path)
fileList = sorted(fileList)
print(fileList)
n = 0
for i in fileList:
  
    oldname = path + os.sep + fileList[n]   
    idx = int(fileList[n].split('.')[0])
    newname = path + os.sep + str(idx+900).zfill(6) + '.png'
    
    os.rename(oldname, newname)   
    print(idx, oldname,'===>',newname)
    
    n += 1

