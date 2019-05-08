import scipy.ndimage as ndi
import scipy
import numpy
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import time
from math import pi
import cv2
import matplotlib.pyplot as plt

mod = SourceModule("""
  __device__ float get_gradient_sum_row(unsigned char *image, float *kernel, int i_x, int k_x, int idy, int kernel_size, int cols, int idx){
    int i_y, index, k_index;
    float sum=0;
    for (int ky=0; ky < kernel_size; ky++) {
        i_y = idy - kernel_size/2 + ky;
        if (i_y < 0 || i_y >= cols) continue;
        index = i_x * cols + i_y;
        k_index = k_x * kernel_size + ky;
        
        float value_to_add = (int) image[index] * kernel[k_index];
        if(idx == 1 && idy ==1){
            //printf("i_value- %d, value- %f, i_x- %d, i_y- %d, k_index- %d, k_value- %f \\n", image[index], value_to_add, i_x, i_y, k_index, kernel[k_index]);
        }
        sum += value_to_add;
    }
    return sum;
  }

  __global__ void get_gradients_gpu_kernel(unsigned char *image, float *kernel_x, float *kernel_y, int kernel_size, int rows, int cols, float *gradients_x, float *gradients_y) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx >= rows || idy >= cols) {
        return ;
    }

    int target_index = idx * cols + idy;

    int x;

    float sum=0;
    int kx = threadIdx.z % kernel_size;
    x = idx - kernel_size/2 + kx;
    if (x < 0 || x>= rows) return;
    
    
    
    if(threadIdx.z / kernel_size == 0){
        sum = get_gradient_sum_row(image, kernel_x, x, kx, idy, kernel_size, cols, idx);
        atomicAdd((float*)(gradients_x + target_index), sum);
    } else{
        sum = get_gradient_sum_row(image, kernel_y, x, kx, idy, kernel_size, cols, idx);
        atomicAdd((float*)(gradients_y + target_index), sum);
    }
  }
  """)


def get_gradient_image_gpu(input_mat, kernel_x, kernel_y, kernel_size):
    kernel_x = np.asarray(kernel_x, dtype=np.float32)
    kernel_y = np.asarray(kernel_y, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)

    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    threads_in_block = int(np.sqrt(1024//kernel_size//2))
    grid = (rows // threads_in_block + 1, cols // threads_in_block + 1, 1)
    block = (threads_in_block, threads_in_block, kernel_size * 2)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_x_gpu = gpuarray.to_gpu(kernel_x)
    kernel_y_gpu = gpuarray.to_gpu(kernel_y)
    gradients_x = np.zeros((rows, cols), dtype=np.float32)
    gradients_y = np.zeros((rows, cols), dtype=np.float32)

    get_gradients_gpu_kernel = mod.get_function("get_gradients_gpu_kernel")
    start_gpu_internal_time = time.time()
    print(kernel_x)
    get_gradients_gpu_kernel(input_mat_gpu, kernel_x_gpu, kernel_y_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(gradients_x), cuda.InOut(gradients_y), block=block, grid=grid)
    print("time taken for gpu internally for single row--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return gradients_x, gradients_y


f = 'ema_stone.jpg'
img = Image.open(f).convert('L')                                          #grayscale
imgdata = numpy.array(img, dtype = float)
sigma = 2.2
image = ndi.filters.gaussian_filter(imgdata, sigma)
# image = np.arange(1, 10).reshape((3, 3))
sobel_x = [[-1.,0.,1.],
           [-2.,0.,2.],
           [-1.,0.,1.]]
kernel_x = np.transpose(sobel_x).copy()
sobel_y = [[-1.,-2.,-1.],
           [0.,0.,0.],
           [1.,2.,1.]]
kernel_y = np.transpose(sobel_y).copy()

imagesToPlot = []

imagesToPlot.append((imgdata.copy(), 'original gray'))

imagesToPlot.append((image.copy(), 'gaussian'))


sobelout = Image.new('L', img.size)                                       #empty image
gradients_x = numpy.array(sobelout, dtype = np.float32)
gradients_y = numpy.array(sobelout, dtype = np.float32)

gradients_x, gradients_y = get_gradient_image_gpu(image, kernel_x, kernel_y, 3)
print(gradients_x[50][70])
sobeloutmag = scipy.hypot(gradients_x, gradients_y)
sobeloutdir = scipy.arctan2(gradients_y, gradients_x)
# print(sobeloutmag)

imagesToPlot.append((sobeloutmag.copy(), 'sobeloutmag'))
imagesToPlot.append((sobeloutdir.copy(), 'sobeloutdir'))

for index, d in enumerate(imagesToPlot, start=1):
    plt.subplot(2, 5, index)
    # plt.imshow(cv2.cvtColor(d[0].astype(np.uint8), cv2.COLOR_GRAY2RGB))
    plt.imshow(d[0].astype(np.uint8), cmap='gray')
    plt.title(d[1])

plt.show()
