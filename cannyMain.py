import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
from skimage import feature
from skimage import io
import matplotlib.image as mpimg


# Generate noisy image of a square
# im = np.zeros((256, 256))
# im[64:-64, 64:-64] = 1
# im[10:50, 10:40] = 1
#
# im = ndi.rotate(im, 15, mode='constant')
# im = ndi.gaussian_filter(im, 4)
# im += 0.2 * np.random.random(im.shape)

im = io.imread('boat.jpg')
edges = np.uint8(feature.canny(im, sigma=1, ) * 255)
# edges[50][150] = True
# print(np.sum(edges))
io.imshow(edges)
io.show()

# im = mpimg.imread('faceWoman.jpg')
# gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
# print(im)
# blurred = cv.GaussianBlur(gray, (9, 9), 0)
# # blurredColor = cv.cvtColor(blurred, cv.COLOR_GRAY2RGB)
# # Compute the Canny filter for two values of sigma
# edges1 = feature.canny(blurred)
# edges2 = feature.canny(blurred, sigma=2)
#
# print(edges1)
#
# # display results
# # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
# #                                     sharex=True, sharey=True)
#
# fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
#                                     sharex=True, sharey=True)
#
# ax1.imshow(blurred)
# ax1.axis('off')
# ax1.set_title('Main Image', fontsize=12)
#
# # ax2.imshow(blurred, cmap=plt.cm.gray)
# # ax2.axis('off')
# # ax2.set_title('Wide bound', fontsize=20)
#
# ax3.imshow(edges1, cmap=plt.cm.gray)
# ax3.axis('off')
# # ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)
# ax3.set_title('Edges', fontsize=12)
#
# fig.tight_layout()
#
# plt.show()
