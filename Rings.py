import cv2 as cv
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import queue
import sys
import os
np.set_printoptions


#start_time = time.time()
def threshold(img,thresh):
    img[img > thresh] = 255
    img[img <= thresh] = 0
    return img


def invert(img): 
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j] == 0: 
                img[i,j] = 255
            else:
                img[i,j] = 0
    #after = time.time()
    #timeTaken = after - before
    return img

def threshhold(img, thresh):
     for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i, j] > thresh: 
               img[i,j] = 255
            else: 
               img[i,j] = 0
     return img


def img_hist(img):
    hist = np.zeros(256)
    for i in range(0,img.shape[0]):#loop through the rows
        for j in range(0,img.shape[1]):#loop through the columns
            hist[img[i,j]]+=1
    return hist

def find_thresh(hist):

    max = 0
    large = -1
   

    for i in range(hist.shape[0]):
        if hist[i] > max: 
            max = hist[i]
            large = i
    print('The max value is \n' + str(max))
    print('The value index of the max value is \n' + str(large))
    #print("Elapsed time:",t1_start)
    #print("--- %s seconds ---" % (time.time() - start_time))
    return large - 75


def dilate_img(img, number_rounds):
    before = time.time()
    for k in range(number_rounds): 
        img_1 = img.copy() 
        for i in range(0,img.shape[0]-1):
            for j in range(0,img.shape[1]-1):
                if img [i -1, j-1] == 255:
                   img_1[i,j]= 255
                elif img[i -1,j] == 255: 
                   img_1[i,j]= 255
                elif img[i -1,j + 1] == 255: 
                   img_1[i,j]= 255
                elif img[i,j -1] == 255: 
                   img_1[i,j] = 255
                elif img[i,j +1] == 255:
                   img_1[i,j] = 255
                elif img[i +1,j -1] == 255:
                   img_1[i,j] = 255
                elif img[i +1,j] == 255:
                   img_1[i,j]= 255
                elif img[i +1,j + 1] == 255:
                   img_1[i,j] = 255
        img = img_1
    return img_1


def CCL(img):
   ## should return label image 
       labels_imgs = np.zeros((img.shape[0], img.shape[1]),dtype='uint8')
       curlab = 80
       queue = []
       for i in range(0,img.shape[0]):
           for j in range(0,img.shape[1]):
               if img [i,j] == 255 and labels_imgs[i,j] == 0:
                  labels_imgs[i,j] = curlab
                  queue.append((i,j))
                  while len(queue) != 0:
                    pop = queue.pop(0)
                    y = pop[0]
                    x = pop[1]
                    if img[y-1,x] == 255 and labels_imgs[y-1,x] == 0:
                       labels_imgs[y-1,x] = curlab
                       queue.append((y-1,x))
                    if img[y+1,x] == 255 and labels_imgs[y+1,x] == 0:
                       labels_imgs[y+1,x] = curlab
                       queue.append((y+1,x))
                    if img[y,x-1] == 255 and labels_imgs[y,x-1] == 0:
                       labels_imgs[y,x-1] = curlab
                       queue.append((y,x-1))
                    if img[y,x+1] == 255 and labels_imgs[y,x+1] == 0:
                       labels_imgs[y,x+1] = curlab
                       queue.append((y,x+1))
               curlab=curlab+80
       return labels_imgs

def displayTime(img):

    font = cv.FONT_HERSHEY_SIMPLEX
    img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
    cv.putText(img, 'Time: ' +str(timeTaken)+" sec", (5,210), font, 0.4, (255,255,255), 1, cv.LINE_AA)
    return img


path = 'C:/Users/35383/Desktop/Four year/Semester 2/Computer Vision/Assigmenet/Orings/Oring'
i=1
while True:
    
    #read in an image into memory
    img = cv.imread(path + str(i) + '.jpg',0)
    before = time.time()
    i=(i+1)%15
    if i==0:
        i+=1
   
    #before = time.time()
    hist = img_hist(img)
    thresh = find_thresh(hist)
    thresh_imgs = threshhold(img,thresh)
    #threshold(img,thresh)
    
    imgs = invert(thresh_imgs)
    dilate_imgs = dilate_img(img,5)
    labels_imgs = CCL(dilate_imgs)
    print(labels_imgs)
    after = time.time()
    timeTaken = round(after-before, 2)


    
    #morphology function
    #filter_image(img)
    #CCL function
    #defect detection function
    finalImg = displayTime(img)

 
    cv.imshow('thresholded image 1',imgs)
    cv.imshow('Binary image',dilate_imgs)
    cv.imshow('CCL image ',labels_imgs)
    cv.imshow('processing time image ',finalImg)

    
    plt.plot(hist)
    plt.show()
    
    
    ch = cv.waitKey(100)
    if ch & 0xFF == ord('q'):
        break

