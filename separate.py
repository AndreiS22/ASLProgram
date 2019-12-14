import cv2
import os

target = raw_input("From where to copy?\n")
print target
for file in os.listdir(target):
    if file.endswith('.jpg'):
        print file
        num = int(file[0]) * 1000 + int(file[1]) * 100 + int(file[2]) * 10 + int(file[3])
        letter = target[4]
        if num <= 1330:
            print "case 1\n"
            os.rename(target + "/" + file, "train" + letter + "/" + file)
        else:
            print "case 2\n"
            os.rename(target + "/" + file, "test" + letter + "/" + file)