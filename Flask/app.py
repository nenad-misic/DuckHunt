from flask import Flask
from flask import request
from flask import send_from_directory

import pyautogui
import matplotlib.pyplot as plt
import matplotlib

import requests
import shutil

import numpy as np
import cv2

import os
import math
import random
from datetime import datetime
import argparse

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array

BIG_WIDTH = 1600
BIG_HEIGHT = 1200
SMALL_WIDTH = 400
SMALL_HEIGHT = 300

DISPLAY_CROP_WIDTH = 750
DISPLAY_CROP_HEIGHT = 500

WIDTH_RATIO = BIG_WIDTH / SMALL_WIDTH
HEIGHT_RATIO = BIG_HEIGHT / SMALL_HEIGHT

APPROX_DISPLAY_WIDTH_ON_SMALL = 125
APPROX_DISPLAY_HEIGHT_ON_SMALL = 80

COLOR_GREEN = (35,255,12)
COLOR_RED = (255,0,0)

def load_image(path):
    #loads image and converts it to RGB
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def load_gray_image(path):
    #loads image and converts it to grayscale
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
def grayscale_loaded_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def display_image(image):
    #shows RGB image
    plt.imshow(image)
def display_gray_image(image):
    #shows grayscaled image
    plt.imshow(image, 'gray')
def resize_image(image, width, height):
    #resizes image to height * width
    return cv2.resize(image, (width,height))
def blur_image(image):
    return cv2.GaussianBlur(image, (5,5), 0)
def do_canny(image, parameter1=30, parameter2=130):
    return cv2.Canny(blur_image(image), parameter1, parameter2, 1)
def cnn_predict_image(model, image):
    image = img_to_array(blur_image(image))
    image = np.expand_dims(image, axis=0)
    images = np.vstack([image])
    predictions = model.predict_classes(images, batch_size=1)
    return predictions[0]
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

#expects grayscale image containing display
#returns tuple (croppedImage,crosshairCoordinates)
def crop_the_display_out_of_image(image):
    #scale the image down to remove the noise as we are only detecting display
    image_small = resize_image(image, SMALL_WIDTH, SMALL_HEIGHT)
    
    #find threshold for canny edge detector using OTSU method
    high_thresh, thresh_im = cv2.threshold(image_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    
    #apply canny edge detector to given image
    canny = do_canny(image_small, low_thresh,high_thresh)

    #zero out the borders
    borderLen = 5
    lenx, leny = canny.shape

    canny[0:borderLen,0:leny] = 0
    canny[lenx-borderLen:lenx,0:leny] = 0
    canny[0:lenx,0:borderLen] = 0
    canny[0:lenx,leny-borderLen:leny] = 0

    #find contours from image returned by canny edge detector
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    potential_contours = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #check if contour is potential for display
        if(w > APPROX_DISPLAY_WIDTH_ON_SMALL and h > APPROX_DISPLAY_HEIGHT_ON_SMALL):
            potential_contours.append([x,y,w,h])

    if(len(potential_contours) == 1):
        #we probably found the display contour
        x,y,w,h = potential_contours[0]
        
        #scale parameters up because canny is applied to reduced image
        xd, yd, wd, hd = int(WIDTH_RATIO*x), int(HEIGHT_RATIO*y), int(WIDTH_RATIO*w), int(HEIGHT_RATIO*h)
        
        #find central pixel on image
        central = (int(image.shape[1]/2), int(image.shape[0]/2))

        #crop the display out of the image
        crop_img = image[yd:yd+hd, xd:xd+wd]
        #find the coordinates of central pixel on display
        cropped_central = (int(central[0]-xd), int(central[1]-yd))
        
        return crop_img, cropped_central
    else:
        #we found more than 1 potential contour
        #log the value, so we know how to debug it :)
        print(len(potential_contours))

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(100,100,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    return model


image_url = "http://192.168.0.10:8080/photo.jpg"
model = create_model()
model.load_weights("../Classifiers/best_weights_cnn.hdf5")
app = Flask(__name__, static_url_path='')

@app.route('/')
def servePage():
    return send_from_directory('static', 'index.html')

@app.route('/shoot')
def shoot():
    resp = requests.get(image_url, stream=True)
    resp.raw.decode_content = True

    x = np.fromstring(resp.raw.read(), dtype='uint8')

    image = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
    grayscale_image = grayscale_loaded_image(image)
    resized_image = resize_image(grayscale_image, BIG_WIDTH, BIG_HEIGHT)
    blurred = blur_image(resized_image)
    crop_image, central = crop_the_display_out_of_image(blurred)
    crop_image_resize = resize_image(crop_image,DISPLAY_CROP_WIDTH,DISPLAY_CROP_HEIGHT)
    resize_ratio_width = DISPLAY_CROP_WIDTH / crop_image.shape[1]
    resize_ratio_height = DISPLAY_CROP_HEIGHT / crop_image.shape[0]
    central_resized = (int(central[0] * resize_ratio_width), int(central[1] * resize_ratio_height))
    shooting_rectangle = crop_image_resize[central_resized[1] - 50: central_resized[1] + 50,central_resized[0] - 50 : central_resized[0] + 50]
    display_gray_image(shooting_rectangle)

    prediction = cnn_predict_image(model, shooting_rectangle)

    screen_width,screen_height = 800,500
    transfer_ratio_width = DISPLAY_CROP_WIDTH / screen_width
    transfer_ratio_height = DISPLAY_CROP_HEIGHT / screen_height
    central_transfered = (int(central[0] * transfer_ratio_width), int(central[1] * transfer_ratio_height))
    print(central_transfered)
    pyautogui.moveTo(central_transfered[0]+30, central_transfered[1]+50, duration=0.5)
    
    if prediction == 0:
        pyautogui.click()
    

    return 'bird' if prediction==0 else 'not bird'

if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=False)