import cv2
import os
import glob
import shutil

lie = 0
olie = 0
lie_path = './lie/'
olie_path = './olie/'


for img in sorted(glob.glob("./user15-43/*.png")):
    cv_img = cv2.imread(img)
    print(img)
    cv2.imshow(str(img), cv_img)
    while True:
            if cv2.waitKey(1) & 0xFF == ord('o'):
                lie += 1
                print('lie', lie)
                print('name', img)
                cv2.destroyAllWindows()
                shutil.move(img, lie_path)
                break  

            if cv2.waitKey(1) & 0xFF == ord('x'):
                olie += 1
                print('olie', olie)
                print('name', img)
                cv2.destroyAllWindows()
                shutil.move(img, olie_path)
                break  

print('lie', lie, 'olie', olie)
