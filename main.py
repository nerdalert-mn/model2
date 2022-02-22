from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threading import Thread
import action

import os

def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]

def on_detect(frame):
    cv2.imwrite("./detection.jpg", frame) 
    action.send_notification()

def mamon_videoFightModel2(tf,wight='mamonbest947oscombo.hdfs'):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30 , return_sequences= True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model


model = mamon_videoFightModel2(tf,wight='mamonbest947oscombo.hdfs')

cap = cv2.VideoCapture('rtsp://admin:@192.168.86.200/H265?ch=1&subtype=0')
i = 0
frames = np.zeros((30, 160, 160, 3), dtype=np.float)
old = []
j = 0
# print('Violence detected\n'*2)
while(True):
    ret, frame = cap.read()
    image = frame
    if i > 29:
        ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
        ysdatav2[0][:][:] = frames
        predaction = pred_fight(model,ysdatav2,acuracy=0.96)
        if predaction[0] == True:
            print('violence detected')
            thread = Thread(target = on_detect, args = [image])
            thread.start()
            # cv2.imshow('video', frame)
            # print('Violence detected')
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # vio = cv2.VideoWriter("./trimmed_videos/output-"+str(j)+".avi", fourcc, 10.0, (fwidth,fheight))
            # for frameinss in old:
            #     vio.write(frameinss)
            # vio.release()
        else:
            print('no violence')
        i = 0
        j += 1
        frames = np.zeros((30, 160, 160, 3), dtype=np.float)
        old = []
        try:
            frame = cv2.resize(frame,(960,540))
        except Exception as e:
            print("LOL")
    else:
        try:
            frm = resize(frame,(160,160,3))
            old.append(frame)
            fshape = frame.shape
            fheight = fshape[0]
            fwidth = fshape[1]
            frm = np.expand_dims(frm,axis=0)
            if(np.max(frm)>1):
                frm = frm/255.0
            frames[i][:] = frm
            frame = cv2.resize(frame,(960,540))
            i+=1
        except Exception as e:
            print("Video ended")
            cap.release()
            cv2.destroyAllWindows()
            
    cv2.imshow('video', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
        
