#Importing the libraries
import cv2
import numpy as np

def Canny(image):
    
    #Color to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    poly = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#load the image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

#Perform Canny on the loaded image
canny = Canny(lane_image)

#Bitwise conversion
cropped_image = region_of_interest(canny)

cv2.imshow('result', cropped_image)
cv2.waitKey(0)
