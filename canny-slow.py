import scipy.ndimage as ndi
import scipy
import numpy
import numpy as np
from PIL import Image
import math
from math import pi
import cv2
import matplotlib.pyplot as plt

sigma = 2.2

f = 'ema_stone.jpg'
img = Image.open(f).convert('L')                                          #grayscale
imgdata = numpy.array(img, dtype = float)                                 
G = ndi.filters.gaussian_filter(imgdata, sigma)                           #gaussian low pass filter

imagesToPlot = []

imagesToPlot.append((imgdata.copy(), 'original gray'))

imagesToPlot.append((G.copy(), 'gaussian'))


sobelout = Image.new('L', img.size)                                       #empty image
gradx = numpy.array(sobelout, dtype = np.float32)
grady = numpy.array(sobelout, dtype = np.float32)

sobel_x = [[-1.,0.,1.],
           [-2.,0.,2.],
           [-1.,0.,1.]]
sobel_y = [[-1.,-2.,-1.],
           [0.,0.,0.],
           [1.,2.,1.]]

width = img.size[1]
height = img.size[0]

#calculate |G| and dir(G)

for x in range(1, width-1):
    for y in range(1, height-1):
        px = (sobel_x[0][0] * G[x-1][y-1]) + (sobel_x[0][1] * G[x][y-1]) + \
             (sobel_x[0][2] * G[x+1][y-1]) + (sobel_x[1][0] * G[x-1][y]) + \
             (sobel_x[1][1] * G[x][y]) + (sobel_x[1][2] * G[x+1][y]) + \
             (sobel_x[2][0] * G[x-1][y+1]) + (sobel_x[2][1] * G[x][y+1]) + \
             (sobel_x[2][2] * G[x+1][y+1])

        py = (sobel_y[0][0] * G[x-1][y-1]) + (sobel_y[0][1] * G[x][y-1]) + \
             (sobel_y[0][2] * G[x+1][y-1]) + (sobel_y[1][0] * G[x-1][y]) + \
             (sobel_y[1][1] * G[x][y]) + (sobel_y[1][2] * G[x+1][y]) + \
             (sobel_y[2][0] * G[x-1][y+1]) + (sobel_y[2][1] * G[x][y+1]) + \
             (sobel_y[2][2] * G[x+1][y+1])
        gradx[x][y] = px
        grady[x][y] = py

sobeloutmag = scipy.hypot(gradx, grady)
sobeloutdir = scipy.arctan2(grady, gradx)

# scipy.misc.imsave('cannynewmag.jpg', sobeloutmag)
# scipy.misc.imsave('cannynewdir.jpg', sobeloutdir)

imagesToPlot.append((sobeloutmag.copy(), 'sobeloutmag'))
imagesToPlot.append((sobeloutdir.copy(), 'sobeloutdir'))


for x in range(width):
    for y in range(height):
        if (sobeloutdir[x][y]<22.5 and sobeloutdir[x][y]>=0) or \
           (sobeloutdir[x][y]>=157.5 and sobeloutdir[x][y]<202.5) or \
           (sobeloutdir[x][y]>=337.5 and sobeloutdir[x][y]<=360):
            sobeloutdir[x][y]=0
        elif (sobeloutdir[x][y]>=22.5 and sobeloutdir[x][y]<67.5) or \
             (sobeloutdir[x][y]>=202.5 and sobeloutdir[x][y]<247.5):
            sobeloutdir[x][y]=45
        elif (sobeloutdir[x][y]>=67.5 and sobeloutdir[x][y]<112.5)or \
             (sobeloutdir[x][y]>=247.5 and sobeloutdir[x][y]<292.5):
            sobeloutdir[x][y]=90
        else:
            sobeloutdir[x][y]=135


# scipy.misc.imsave('cannynewdirquantize.jpg', sobeloutdir)

imagesToPlot.append((sobeloutdir.copy(), 'dirquantize'))


mag_sup = sobeloutmag.copy()

for x in range(1, width-1):
    for y in range(1, height-1):
        if sobeloutdir[x][y]==0:
            if (sobeloutmag[x][y]<=sobeloutmag[x][y+1]) or \
               (sobeloutmag[x][y]<=sobeloutmag[x][y-1]):
                mag_sup[x][y]=0
        elif sobeloutdir[x][y]==45:
            if (sobeloutmag[x][y]<=sobeloutmag[x-1][y+1]) or \
               (sobeloutmag[x][y]<=sobeloutmag[x+1][y-1]):
                mag_sup[x][y]=0
        elif sobeloutdir[x][y]==90:
            if (sobeloutmag[x][y]<=sobeloutmag[x+1][y]) or \
               (sobeloutmag[x][y]<=sobeloutmag[x-1][y]):
                mag_sup[x][y]=0
        else:
            if (sobeloutmag[x][y]<=sobeloutmag[x+1][y+1]) or \
               (sobeloutmag[x][y]<=sobeloutmag[x-1][y-1]):
                mag_sup[x][y]=0

# scipy.misc.imsave('cannynewmagsup.jpg', mag_sup)

imagesToPlot.append((mag_sup.copy(), 'mag_sup'))

m = numpy.max(mag_sup)
th = 0.2*m
tl = 0.1*m


gnh = numpy.zeros((width, height))
gnl = numpy.zeros((width, height))

for x in range(width):
    for y in range(height):
        if mag_sup[x][y]>=th:
            gnh[x][y]=mag_sup[x][y]
        if mag_sup[x][y]>=tl:
            gnl[x][y]=mag_sup[x][y]
# scipy.misc.imsave('cannynewgnlbeforeminus.jpg', gnl)

imagesToPlot.append((gnl.copy(), 'gnl_before_minus'))

gnl = gnl-gnh
# scipy.misc.imsave('cannynewgnlafterminus.jpg', gnl)
# scipy.misc.imsave('cannynewgnh.jpg', gnh)

imagesToPlot.append((gnl.copy(), 'gnl_after_minus'))

imagesToPlot.append((gnh.copy(), 'gnh'))


def traverse(i, j):
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    for k in range(8):
        if gnh[i+x[k]][j+y[k]]==0 and gnl[i+x[k]][j+y[k]]!=0:
            gnh[i+x[k]][j+y[k]]=1
            traverse(i+x[k], j+y[k])

for i in range(1, width-1):
    for j in range(1, height-1):
        if gnh[i][j]:
            gnh[i][j]=1
            traverse(i, j)


# scipy.misc.imsave('cannynewout.jpg', gnh)

imagesToPlot.append((gnh.copy(), 'final'))

for index, d in enumerate(imagesToPlot, start=1):
    plt.subplot(2, 5, index)
    # plt.imshow(cv2.cvtColor(d[0].astype(np.uint8), cv2.COLOR_GRAY2RGB))
    plt.imshow(d[0].astype(np.uint8), cmap='gray')
    plt.title(d[1])

plt.show()
