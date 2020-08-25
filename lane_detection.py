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

def average_slope_Intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    #print(left_fit_avg)
    #print(right_fit_avg)
    left_line = make_coord(image, left_fit_avg)
    right_line = make_coord(image, right_fit_avg)
    return np.array([left_line, right_line])

#load the image
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

#Perform Canny on the loaded image
canny = Canny(lane_image)

#Bitwise conversion
cropped_image = region_of_interest(canny)

cv2.imshow('result', cropped_image)
cv2.waitKey(0)
