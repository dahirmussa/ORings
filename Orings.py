##project by Dylan Hallissey


import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
np.set_printoptions


def thresshold(img, thresh):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > thresh:
                img[i, j] = 255
            else:
                img[i,j] = 0
    return img




def img_hist(img):
    hist = np.zeros(256)
    for i in range(0,img.shape[0]):#loop through the rows
        for j in range(0,img.shape[1]):#loop through the columns
            hist[img[i,j]]+=1
    return hist


def invert(img):
       before = time.time()
       for i in range (0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] == 0:
                img[i,j] = 255
            else:
                img[i,j] = 0
        after = time.time()
        timeTaken = after - before
        font = cv.FONT_HERSHEY_SIMPLEX
       cv.putText(img, 'Time: ' +str(timeTaken)+" sec", (5,210), font, 0.4, (255,255,255), 1, cv.LINE_AA)

       return img

def find_thresh(hist):
    max = 0
    max2 = -1
    for i in range(hist.shape[0]):
        if hist[i] > max:
            max = hist[i]
            max2 = i
    print('the value for the max is:' + str(max))
    print('the index value for the max is:' + str(max2))
    
    return max2-100




def dilate(img, number_rounds):

    for k in range(number_rounds):
        img_copy = img.copy()
        for i in range(1,img.shape[0]-1):#loop through the rows
           for j in range(1,img.shape[1]-1):#loop through the columns
                if img[i-1,j -1]==255:
                   img_copy[i,j] = 255
                elif img[i-1,j]==255:
                   img_copy[i,j] = 255
                elif img[i-1,j+1]==255:
                   img_copy[i,j] = 255
                elif img[i,j-1]==255:
                   img_copy[i,j] = 255
                elif img[i,j+1]==255:
                   img_copy[i,j] = 255
                elif img[i+1,j-1]==255:
                   img_copy[i,j] = 255
                elif img[i+1,j]==255:
                   img_copy[i,j] = 255
                elif img[i+1,j+1]==255:
                   img_copy[i,j] = 255
        img = img_copy

    return img

        
        
def CCL(img):
    labeled_img = np.zeros((img.shape[0], img.shape[1]),dtype='uint8')
    curlab = 1
    queue = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] == 255 and labeled_img[i,j] == 0:
                labeled_img[i,j] = curlab
                queue.append((i, j))
                while len(queue) != 0:
                   pop = queue.pop(0)
                   y = pop[0]
                   x = pop[1]
                   if img[y-1,x] ==255 and labeled_img [y-1,x] ==0:
                       labeled_img[y-1, x] = curlab
                       queue.append((y-1, x))
                   
            #curlab+=1
            curlab=curlab=80
    return labeled_img          


path = 'C:/Users/35383/Desktop/Four year/Semester 2/Computer Vision/Assigmenet/Orings/Oring'
i=1
while True:
    
    #read in an image into memory
    img = cv.imread(path + str(i) + '.jpg',0)
    i=(i+1)%15
    if i==0:
        i+=1


    hist = img_hist(img)
    thresh = find_thresh(hist)    
    thresshold(img,thresh)
    thresh_imgs = thresshold(img, thresh)
    img = invert(img)
    dilated_img = dilate(img,5)
    labeled_img = CCL (dilated_img)
    print(labeled_img)

    #morphology function
    #CCL function
    #defect detection function
    #dilated_img, 
    
    cv.imshow('thresholded image 1',thresh_imgs)
    cv.imshow('dilated image 1 ', dilated_img)
    cv.imshow('labeled image 1 ', labeled_img)

    plt.plot(hist)
    plt.show()

    #ch = cv.waitKey()
    ch = cv.waitKey(100)
    if ch & 0xFF == ord('q'):
        break