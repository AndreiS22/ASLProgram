import cv2
import os

for folder in os.listdir('source'):
    name = ''
    if len(folder) > 5:
        for i in range(5):
            name += folder[i]
    if name == 'train':
        letter = folder[5]
        for imgurl in os.listdir('source/' + folder):
            # if imgurl.endswith('.jpg'):
            #     path = 'source/' + folder + '/' + imgurl
            #     img = cv2.imread(path)
            #     os.remove(path)
            #     newpath = 'source/' + folder + '/' + letter + imgurl
            #     cv2.imwrite(newpath, img)
            if imgurl.endswith('.lst'):
                with open('source/' + folder + '/' + imgurl, 'w') as f:
                    print f