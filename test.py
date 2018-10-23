# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import json
import codecs

import keras
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.models import model_from_json

BATCH_SIZE = 128
N_CLASSES = 3095
LR = 0.001
N_EPOCHS = 50
N_UNITS = 128
IMG_SIZE = 100

model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)
model.load_weights("models/weights-improvement-26-0.00.hdf5")
model.summary()
with open("models/model.json", "w") as json_file:
    json_file.write(model.to_json())

model2 = model_from_json(open('models/model.json', 'r').read())
model2.load_weights("models/weights-improvement-26-0.00.hdf5")

def transform(img):
    height, width = img.shape
    desired_size = int(height*1.1)
    delta_w = desired_size - width
    delta_h = desired_size - height
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))

img = cv2.imread('test.png')
cv2.imshow('img', img)
cv2.waitKey(0)

labels = json.loads(codecs.open("labels.txt", "r", "utf-8").read())
bboxes = [(66,488,146,613),
        (147,488,225,602),
        (238,495,292,596),
        (319,493,371,597),
        (386,483,469,601),
        (479,496,530,595),
        (562,491,610,594),
        (630,487,699,595),
        (720,494,768,591),
        (800,491,848,590),
        (874,485,938,594),
        (952,483,1019,596),
        (1027,488,1103,593),
        (1104,481,1183,597),
        (1186,477,1262,594)]

file = codecs.open("result.txt", "w", "utf-8")

for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    subimg = np.copy(img[ymin:ymax,xmin:xmax])
    subimg_gray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
    retval2, subimg_binary = cv2.threshold(subimg_gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    subimg_transformed = transform(subimg_binary)
    subimg_final = cv2.resize(255 - subimg_transformed, (100, 100))
    cv2.imshow('segment', subimg)
    cv2.imshow('segment2', subimg_final)
    
    subimg_rgb = cv2.cvtColor(subimg_final, cv2.COLOR_GRAY2RGB)
    X = subimg_rgb.astype(np.float64)
    X *= 1./255
    X = np.array([X])
    Y = model2.predict(X)
    Y_label = Y[0].argsort()[-3:][::-1]
    for i in range(len(Y_label)):
        idx_int = Y_label[i]
        idx_str = str(idx_int)
        s = "Top {}: {} - score {}\n".format(i, idx_str, Y[0][idx_int])
        # file.write(s)
        print(s)
    print('Done')
    file.write('=======================\n')
    cv2.waitKey(0)
    # break

file.close()
cv2.destroyAllWindows()
exit(1)
