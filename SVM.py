import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_hog() : 
    winSize = (200, 200)
    blockSize = (100, 100)
    blockStride = (50, 50)
    cellSize = (50, 50)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR



def svmInit(C=16, gamma=0.9):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, digits, samples, labels):
    fpr = np.zeros(26)
    fnr = np.zeros(26)
    predictions = svmPredict(model, samples)
    predict_labels = []
    for x in predictions:
        predict_labels.append(chr(x))
    accuracy = (labels == predictions).mean()
    print predictions
    print 'Percentage Accuracy: %.2f %%' % (accuracy*100)

    confusion = np.zeros((26, 26), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i) - 86, int(j) - 86] += 1
    for i in labels:
        actualno = 0
        #print "letter = " + chr(i)
        for j in predictions:
            if int(i) - 86 != int(j) - 86:
         #       print "predict = " + chr(j), confusion[int(i) - 86, int(j) - 86]
                actualno += confusion[int(i) - 86, int(j) - 86]
        #print "actualno is " + str(actualno)
        if actualno > 0:
            fpr[int(i) - 86] = confusion[int(i) - 86, int(i) - 86] / actualno
    print 'confusion matrix:'
    print confusion
    print fpr
    print fnr

    # vis = []
    # for img, flag in zip(digits, predictions == labels):
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     if not flag:
    #         img[...,:2] = 0
        
    #     vis.append(img)
    # return mosaic(25, vis)

if __name__ == '__main__':
    
    labels = []
    feats = [] 
    # Load data.
    for file in ['../data/HOG/pos']:
       for img in os.listdir(file):
            path = str(file) + '/' + str(img)
            if path.endswith('.jpg'):
                # create lables vector for each image found in the folder
                filename = path.split('/')[4]
                name = filename.split('.')[0]
                letter = name[0]
                labels.append(letter)
        
                # create features vector for each image found in the folder
                img = cv2.imread(path)
                hog = get_hog()
                feats.append(hog.compute(img))
    
    feats = np.squeeze(feats)
    
    # split data and train SVM
    feats_train = []
    labels_train = []
    feats_test = []
    labels_test = []
    index = 0
    for file in ['../data/HOG/pos']:
       for img in os.listdir(file):
            path = str(file) + '/' + str(img)
            if path.endswith('.jpg'):
                name = path.split('/')[4]
                letter = name[0]
                cif1 = name[1]
                cif2 = name[2]
                cif3 = name[3]
                num = int(cif1)
                if cif2.isdigit():
                     num = num * 10 + int(cif2)
                if cif3.isdigit():
                    num = num * 10 + int(cif3)
                if letter < 'S' and letter != 'L':
                    if num <= 360:
                        feats_train.append(feats[index])
                        labels_train.append(labels[index])
                    else:
                        feats_test.append(feats[index])
                        labels_test.append(labels[index])
                if letter >= 'S':
                    if num <= 90:
                        feats_train.append(feats[index])
                        labels_train.append(labels[index])
                    else:
                        feats_test.append(feats[index])
                        labels_test.append(labels[index])
                if letter == 'L':
                    if num <= 270:
                        feats_train.append(feats[index])
                        labels_train.append(labels[index])
                    else:
                        feats_test.append(feats[index])
                        labels_test.append(labels[index])
                index += 1

    model = svmInit()

    asciilabels_train = []
    asciilabels_test = []
    for x in labels_train:
        asciilabels_train.append(ord(x))
    for x in labels_test:
        asciilabels_test.append(ord(x))

    asciilabels_train = np.array(asciilabels_train)
    asciilabels_test = np.array(asciilabels_test)
    feats_train = np.array(feats_train)
    feats_test = np.array(feats_test)

    model = svmTrain(model, feats_train, asciilabels_train)
    vis = svmEvaluate(model, feats_test, feats_test, asciilabels_test)

    print 'done'

    #model = svm.SVC(kernel='rbf', C=12.5,gamma=0.50625)
    # model.fit(feats_train, asciilabels_train)
    
    # predicted = model.predict(feats_test)
    # predicted.reshape(-1, 1)
    # asciilabels_test.reshape(-1, 1)
    # print model.score(predicted, asciilabels_test)

    # # x_min, x_max = feats_train[:, 0].min() - 1, feats_train[:, 0].max() + 1
    # # y_min, y_max = feats_train[:, 1].min() - 1, feats_train[:, 1].max() + 1
    # # h = (x_max / x_min)/100
    # # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    # # np.arange(y_min, y_max, h))
    # # plt.subplot(1, 1, 1)
    # # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Z = Z.reshape(xx.shape)
    # # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # # plt.scatter(feats_train[:, 0], feats_train[:, 1], c=y, cmap=plt.cm.Paired)
    # # plt.xlabel('Sepal length')
    # # plt.ylabel('Sepal width')
    # # plt.xlim(xx.min(), xx.max())
    # # plt.title('SVC with linear kernel')
    # # plt.show()

    # print predicted