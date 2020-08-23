//Importing the libraries
import cv2
import numpy as np

def Canny(image):
    
    //Color to Gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    // Blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

//load the image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

// Perform Canny on the loaded image
canny = Canny(lane_image)

cv2.imshow('result', canny)
cv2.waitKey(0)
