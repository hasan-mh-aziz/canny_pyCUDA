import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('Two_lane_city_streets.jpg')
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(im, (9, 9), 0)
# img = np.asarray(im, dtype=np.uint8)
edges = cv2.Canny(im, 25, 255)
print(blurred)
plt.subplot(121), plt.imshow(im, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blurred, cmap='gray')
plt.show()
