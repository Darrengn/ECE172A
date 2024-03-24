import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2

def meanFilter(img, win_size):
    filter = np.ones((win_size,win_size)) / win_size / win_size
    print(np.shape(img))
    out = signal.convolve(img, filter, mode='same')
    # pad = int((win_size-1)/2)
    # padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # output = np.zeros(np.shape(img),dtype=int)
    # for x in range(np.shape(img)[0]):
    #     for y in range(np.shape(img)[1]):
    #         vals = []
    #         for i in range(win_size):
    #             for j in range(win_size):
    #                 vals.append(padded_img[x+i,y+j])
    #         output[x,y] = int(np.mean(vals))
    #         print((x,y))
    return out.astype(np.uint8)

def medianFilter(img, win_size):
    pad = int((win_size-1)/2)
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros(np.shape(img),dtype=int)
    for x in range(np.shape(img)[0]):
        for y in range(np.shape(img)[1]):
            vals = []
            for i in range(win_size):
                for j in range(win_size):
                    vals.append(padded_img[x+i,y+j])
            output[x,y] = int(np.median(vals))
            
    return output.astype(np.uint8)

def computeNormGrayHistogram(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist = np.zeros(32)
    for i in range(np.shape(grey)[0]):
        for j in range(np.shape(grey)[1]):
            val = int(grey[i,j] / 8)
            hist[val] = hist[val] + 1
    sum = np.sum(hist)
    hist = hist / sum
    return hist


img = cv2.imread('mural.jpg')
noise1 = cv2.imread('mural_noise1.jpg', cv2.IMREAD_GRAYSCALE)
noise2 = cv2.imread('mural_noise2.jpg', cv2.IMREAD_GRAYSCALE)
temp = cv2.imread('template.jpg')
# out = cv2.matchTemplate(img, temp, cv2.TM_CCORR_NORMED)

# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(out)
# print(max_loc)
# cv2.rectangle(img, max_loc, (max_loc[0] + 100, max_loc[1] + 100), 255, 2)

# cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img1', 2400, 1082)
# cv2.imshow('img1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# mean5 = meanFilter(img,5)

# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# data = computeNormGrayHistogram(img)
# data1 = computeNormGrayHistogram(noise1)
# data2 = computeNormGrayHistogram(noise2)
# plt.bar(np.linspace(0, 31, 32), data, color='blue', alpha=0.7)
# plt.title('Greyscale Histogram of Mural')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()

# plt.bar(np.linspace(0, 31, 32), data1, color='blue', alpha=0.7)
# plt.title('Greyscale Histogram of Noise 1')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()

# plt.bar(np.linspace(0, 31, 32), data2, color='blue', alpha=0.7)
# plt.title('Greyscale Histogram of Noise 2')
# plt.xlabel('Bin')
# plt.ylabel('normalized value')
# plt.show()


# mean81 = meanFilter(noise2,81)

# hist_mean81 = cv2.calcHist([mean81], [0], None, [256], [0, 256])

median5 = medianFilter(img,5)

hist_median5 = cv2.calcHist([median5], [0], None, [256], [0, 256])

# median81 = medianFilter(img,81)

# hist_median81 = cv2.calcHist([median81], [0], None, [256], [0, 256])

# plt.bar(range(256), hist.flatten(), color='blue', alpha=0.7)
# plt.title('Noise 2 Mean 5x5 Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img1', 2400, 1082)
# cv2.imshow('img1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# plt.bar(range(256), hist_mean81.flatten(), color='blue', alpha=0.7)
# plt.title('Noise 2 Mean 81x81 Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img2', 2400, 1082)
# cv2.imshow('img2',mean81)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.bar(range(256), hist_median5.flatten(), color='blue', alpha=0.7)
plt.title('Noise 2 Median 5x5 Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img3', 2400, 1082)
# cv2.imshow('img3',median5)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.bar(range(256), hist_median81.flatten(), color='blue', alpha=0.7)
# plt.title('Noise 2 Median 81x81 Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('img4', 2400, 1082)
# cv2.imshow('img4',median81)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
