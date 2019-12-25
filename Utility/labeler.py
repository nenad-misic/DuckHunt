import numpy as np
import cv2
import os
def load_image(path):
    #loads image and converts it to grayscale
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def load_gray_image(path):
    #loads image and converts it to grayscale
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
def display_image(image):
    #shows grayscaled image
    plt.imshow(image)
def display_gray_image(image):
    #shows grayscaled image
    plt.imshow(image, 'gray')
def get_image_size(image):
    #returns tuple (h,w)
    return ipos.shape
def resize_image(image, height, width):
    #resizes image to height * width
    return cv2.resize(image, (height,width))
def blur_image(image):
    return cv2.GaussianBlur(image, (5,5), 0)
def do_canny(image, parameter1=30, parameter2=130):
    return cv2.Canny(blurred, parameter1, parameter2, 1)
prep_dir = 'Dataset2/Ducks_processed_label/'
finish_dirb = 'Dataset2/Ducks_processed_labeltest_bird/'
finish_dirn = 'Dataset2/Ducks_processed_labeltest_notbird/'
i = 0
for img_name in os.listdir(prep_dir):
    i+=1
    img_path = os.path.join(prep_dir, img_name)
    image = load_gray_image(img_path)

    cv2.imshow('image', resize_image(image, 800, 800))
    k = cv2.waitKey(0)
    if k == ord(','):         # NOT A BIRB
        img_path_save = os.path.join(finish_dirn, img_name)
    elif k == ord('.'): # A BIRB
        img_path_save = os.path.join(finish_dirb, img_name)
    else:
        continue
    print(str(i) + "/6174")
    cv2.imwrite(img_path_save, image)
    cv2.destroyAllWindows()