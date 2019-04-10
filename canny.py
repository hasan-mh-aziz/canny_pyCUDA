import numpy as np
import cv2 as cv
from scipy import ndimage as ndi
from skimage import feature
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

actuallImg = mpimg.imread('Two_lane_city_streets.jpg')
grayImg = cv.imread('SM-N950U1_0008_eyelidFlashOff_2018-05-29-12-58-02-PM.jpg')
print(actuallImg)
# plt.imshow(actuallImg)
# plt.show()
blue = np.asarray(actuallImg[:, :, 2], dtype=np.int16)
green = np.asarray(actuallImg[:, :, 1], dtype=np.int16)
red = np.asarray(actuallImg[:, :, 0], dtype=np.int16)

img = np.maximum(0, red - (blue + green)/2)
avgRed = np.average(img[img != 0])
maxImg = np.amax(img)
maxImgIndex = np.unravel_index(np.argmax(img, axis=None), img.shape)
# print(img)
img = np.maximum(0, img - maxImg/5*2 - avgRed)
# img = np.maximum(0, img - avgRed)
print(maxImgIndex, ':', maxImg)
print('average red: ', avgRed)
maxImg = np.amax(img)
maxImgIndex = np.unravel_index(np.argmax(img, axis=None), img.shape)
print(maxImgIndex, ':', maxImg)
# for row in range(0, len(img)):
#     for col in range(0, len(img[0])):
#         if img[row][col] < maxImg - 30:
#             img[row][col] = 0

print(img)
img = np.asarray(img, dtype=np.uint8)
print(img)
edges = cv.Canny(img, avgRed/3*2, maxImg/4)
# edges = feature.canny(img)

greenImg = np.maximum(0, green - (blue + red)/2)
blueImg = np.maximum(0, blue - (green + red)/2)
red = np.asarray(red, dtype=np.uint8)
blue = np.asarray(blue, dtype=np.uint8)
green = np.asarray(green, dtype=np.uint8)
# redEdges = feature.canny(red, sigma=3)
blueEdges = feature.canny(blueImg)
avgGreen = np.average(greenImg[greenImg != 0])
greenImg = np.maximum(0, greenImg - avgGreen)
greenEdges = feature.canny(greenImg, sigma=3)

redModified = np.logical_and(edges, np.logical_not(greenImg))
redModified = np.logical_and(redModified, np.logical_not(blueImg))
plt.subplot(231), plt.imshow(actuallImg, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(img)
plt.subplot(233), plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(greenImg)
plt.title('Red Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(greenEdges)
plt.title('Blue Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(redModified)
plt.title('remaining Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
