

import cv2
import numpy as np
import os
def resize_image(image, width, height):
    #resizes image to height * width
    return cv2.resize(image, (width,height))

if __name__ == '__main__' :
 
    
    prep_dir = 'Dataset4/Ducks_processed_label_bird/'
    # Read image
    for img_name in os.listdir(prep_dir):
        img_path = os.path.join(prep_dir, img_name)
        im = cv2.imread(img_path)

        # Select ROI
        fromCenter = False
        r = cv2.selectROI(im, fromCenter)
        
        # Crop image
        #f1.write(img_name + "," + str(int(r[1])) + "," + str(int(r[1]+r[3])) + "," + str(int(r[0])) + "," + str(int(r[0]+r[2])) + "\n")
        print(str(int(r[1])) + "," + str(int(r[1]+r[3])) + "," + str(int(r[0])) + "," + str(int(r[0]+r[2])))
        # Select ROI
        fromCenter = False
        r = cv2.selectROI(im, fromCenter)
        
        # Crop image
        #f2.write(img_name + "," + str(int(r[1])) + "," + str(int(r[1]+r[3])) + "," + str(int(r[0])) + "," + str(int(r[0]+r[2])) + "\n")

