import urllib2
import urllib
import cv2
import numpy as np
import os

def get_neg():
    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07935504'   
    neg_image_urls = urllib2.urlopen(neg_images_link).read().decode()
    pic_num = 1

    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print i
            urllib.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (400, 400))
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print str(e)

def find_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print 'That is one ugly pic! Deleting!'
                        print current_image_path
                        os.remove(current_image_path)
                except Exception as e:
                    print str(e)
    num = 1
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            path = 'neg/' + img
            image = cv2.imread(path)
            name = 'neg/' + str(num) + '.jpg'
            print name
            cv2.imwrite(name, image)
            num += 1

def create_pos_n_neg():
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'neg' and img.endswith('.jpg'):
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
            # elif file_type == 'pos':
            #     line = file_type+'/'+img+' 1 0 0 50 50\n'
            #     with open('info.dat','a') as f:
            #         f.write(line)
            

#get_neg()
#find_uglies()
create_pos_n_neg()