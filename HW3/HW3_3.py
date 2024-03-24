import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2


img = cv2.imread('lane.png', cv2.IMREAD_GRAYSCALE)

smoothing = 1/159 * np.array([
    [2,4,5,4,2],
    [4,9,12,9,4],
    [5,12,15,12,5],
    [4,9,12,9,4],
    [2,4,5,4,2]
])
kx = np.array([
    [-1, 0, 1],
    [-2,0,2],
    [-1, 0, 1]
])
ky = kx.T

smooth = signal.convolve(img, smoothing, mode='valid')
x_grad = signal.convolve(smooth, kx, mode='valid')
y_grad = signal.convolve(smooth, ky, mode='valid')
plt.imshow(smooth,cmap='gray')
plt.show()
mag = (x_grad**2 + y_grad**2)**1/2

# print(mag.dtype)

norm_mag = (mag / np.max(mag) * 255)
norm_mag = norm_mag.astype('uint8')
plt.imshow(norm_mag,cmap='gray')
plt.title('Gradient Magnitude Plot')
plt.show()

# Non-maximal suppression step
nms = np.zeros(np.shape(mag))
# padded_img = np.pad(mag, ((1, 1), (1, 1)), mode='constant', constant_values=0)
for i in range(1,np.shape(mag)[0] - 1):
    for j in range(1,np.shape(mag)[1] - 1):
        if x_grad[i,j] == 0:
            val = 0
        else:
            val = np.arctan(y_grad[i,j]/x_grad[i,j])
        # change dir to one of 4 directions
        if val <= np.pi/8 and val >= -np.pi/8:
            if mag[i,j-1] > mag[i,j] or mag[i,j+1] > mag[i,j]:
                nms[i,j] = 0
            else:
                nms[i,j] = norm_mag[i,j]
        elif val > np.pi/8 and val < 3 * np.pi/8:
            if mag[i+1,j-1] > mag[i,j] or mag[i-1,j+1] > mag[i,j]:
                nms[i,j] = 0
            else:
                nms[i,j] = norm_mag[i,j]
        elif val < -np.pi/8 and val > -3 * np.pi/8:
            if mag[i+1,j+1] > mag[i,j] or mag[i-1,j-1] > mag[i,j]:
                nms[i,j] = 0
            else:
                nms[i,j] = norm_mag[i,j]
        else:
            if mag[i-1,j] > mag[i,j] or mag[i+1,j] > mag[i,j]:
                nms[i,j] = 0
            else:
                nms[i,j] = norm_mag[i,j]

norm_nms = (nms / np.max(nms) * 255)
norm_nms = norm_nms.astype('uint8')

plt.imshow(norm_nms,cmap='gray')
plt.title('Non-Maximal Suppression Plot')
plt.show()
print(np.max(norm_nms))

for i in range(np.shape(mag)[0]):
    for j in range(np.shape(mag)[1]):
        if(norm_nms[i,j] < 40):
            norm_nms[i,j] = 0
        else: 
            norm_nms[i,j] = 255

plt.imshow(norm_nms,cmap='gray')
plt.title('Non-Maximal Suppression Plot With Thresholding')
plt.show()