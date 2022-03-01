import tkinter as tk
import tkinter.messagebox as mgb
import tkinter.font as tkFont

from random_select import copyFile

window = tk.Tk()
window.title('Random_select')
window.geometry('800x400')
window.configure(background='black')

def get_user_and_date():
    user = str(user_entry.get())
    date = str(date_entry.get())

    #print(user, date)
    mgb.showinfo(title = 'Done',
                 message = 'Successfully Done!')

    return user, date


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

user_date = tk.Button(window, text='Start', command=get_user_and_date)
user_date.pack()
window.mainloop()
