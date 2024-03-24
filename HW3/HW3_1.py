'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def computeNormGrayHistogram(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(32)
    for i in range(np.shape(grey)[0]):
        for j in range(np.shape(grey)[1]):
            val = int(grey[i,j] / 8)
            hist[val] = hist[val] + 1
    sum = np.sum(hist)
    hist = hist / sum
    return hist

def computeNormRGBHistogram(img):
    r_hist = np.zeros(32)
    g_hist = np.zeros(32)
    b_hist = np.zeros(32)
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            val = int(img[i,j,0] / 8)
            r_hist[val] = r_hist[val] + 1
            val = int(img[i,j,1] / 8)
            g_hist[val] = g_hist[val] + 1
            val = int(img[i,j,2] / 8)
            b_hist[val] = b_hist[val] + 1
    sum = np.sum(r_hist)
    r_hist = r_hist / sum
    sum = np.sum(g_hist)
    g_hist = g_hist / sum
    sum = np.sum(b_hist)
    b_hist = b_hist / sum
    return np.hstack((np.hstack((r_hist,g_hist)),b_hist))

def adapHistEqual(img, win_size):
    pad = int((win_size-1)/2)
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros(np.shape(img))
    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            rank = 0
            for i in range(win_size):
                for j in range(win_size):
                    if img[x,y] > padded_img[x+i,y+j]:
                        rank = rank + 1
            output[x,y] = rank / win_size / win_size
            print((x,y))
    return output


    

img = cv2.imread('beach.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# for i in range(np.shape(img)[0]):
#         for j in range(np.shape(img)[1]):
#             if img[i,j,0] * 2 > 255:
#                 img[i,j,0] = 255
#             else:
#                 img[i,j,0] = img[i,j,0] * 2
# data = computeNormGrayHistogram(img)

# plt.bar(np.linspace(0, 31, 32), data, color='blue', alpha=0.7)
# plt.title('Greyscale Histogram')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()
# data = computeNormRGBHistogram(img)
# plt.bar(np.linspace(0, 95, 96), data, color='blue', alpha=0.7)
# plt.title('RGB Histogram')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()
# data = computeNormGrayHistogram(cv2.flip(img,1))
# plt.bar(np.linspace(0, 31, 32), data, color='blue', alpha=0.7)
# plt.title('Greyscale Histogram')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()
# data = computeNormRGBHistogram(cv2.flip(img,1))
# plt.bar(np.linspace(0, 95, 96), data, color='blue', alpha=0.7)
# plt.title('RGB Histogram')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()


ahe = adapHistEqual(img, 129)
# he = cv2.equalizeHist(img)
    

# print(data)
# cv2.imshow('img',he)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imshow('img',ahe)
cv2.waitKey(0)
cv2.destroyAllWindows()
