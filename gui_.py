import tkinter as tk
import tkinter.messagebox as mgb
import tkinter.font as tkFont
import time
import os
import shutil
import random

from random_select import copyFile

def show_window():
    user = user_entry.get()
    date = date_entry.get()
    random_select(user, date)
    time.sleep(3)
    mgb.showinfo(title = 'Done',
                 message = 'Successfully Done!')

def is_inp(name):
    return(name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG'])


def copyFile(fileDir, tarDir, select_num=150):
	pathDir = os.listdir(fileDir)
	sample = random.sample(pathDir, select_num)
	print(sample)
	for name in sample:
	   shutil.copyfile(fileDir+name, tarDir+name)

def random_select(user, date):
    user = user
    date = date
    fileDir = '/home/kenchang/projects/activesample_experiment/gui/' + user + '/' + date + '/FN_night/'
    outputDir = './'
    userDir = outputDir + user
    tarDir = userDir + '/' + date + '_select' + '/'
    create_dir = True
    select_num = 150
    if create_dir == True:
        print('tar', tarDir)
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
    print('done')
    print('random', user, date)


if __name__ == "__main__":
    window = tk.Tk()
    window.title('Random_select')
    window.geometry('800x400')
    window.configure(background='black')
    header_label = tk.Label(window, 
                            font = ('microsoft yahei', 32),
                            width = 20,
                            height = 4,
                            text='Random_select')
    header_label.pack()

    user_frame = tk.Frame(window)
    user_frame.pack(side=tk.TOP)
    user_label = tk.Label(user_frame, 
                        font = ('Arial', 12),
                        width = 20,
                        height = 2,
                        text='User(ex:user02)')
    user_label.pack(side=tk.LEFT)
    user_entry = tk.Entry(user_frame)
    user_entry.pack(side=tk.LEFT)

    date_frame = tk.Frame(window)
    date_frame.pack(side=tk.TOP)
    date_label = tk.Label(date_frame, 
                        font = ('Arial', 12),
                        width = 20,
                        height = 2,
                        text='Date(ex:2022-02-20)')
    date_label.pack(side=tk.LEFT)
    date_entry = tk.Entry(date_frame)
    date_entry.pack(side=tk.LEFT)

    result_label = tk.Label(window)
    result_label.pack()

    user_date = tk.Button(window, text='Start', command=show_window)
    user_date.pack()

    window.mainloop()
