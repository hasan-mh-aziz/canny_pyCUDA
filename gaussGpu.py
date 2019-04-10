import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time


mod = SourceModule("""
  __global__ void gauss_gpu_kernel(unsigned char *image, float *kernel, int kernel_size, int rows, int cols, float *blurred_image) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > rows || idy > cols) {
        return ;
    }
    
    int target_index = idx * cols + idy;
    
    int x, y, index, k_index;
    
    float sum=0;
    for (int kx=0; kx < kernel_size; kx++) {
        x = idx - kernel_size/2 + kx;
        if (x < 0) continue;
        for (int ky=0; ky < kernel_size; ky++) {
            y = idy - kernel_size/2 + ky;
            if (y < 0) continue;
            
            index = x * cols + y;
            k_index = kx * kernel_size + ky;
            
            sum += (int) image[index] * kernel[k_index];
        }
    }
    blurred_image[target_index] = sum;
  }
  
  __global__ void gauss_gpu_kernel_v2(unsigned char *image, float *kernel, int kernel_size, int rows, int cols, float *blurred_image) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > rows || idy > cols) {
        return ;
    }
    
    int target_index = idx * cols + idy;
    
    int x, y, index, k_index;
    
    float sum=0;
    int kx = threadIdx.z;
    x = idx - kernel_size/2 + kx;
    if (x < 0) return;
    for (int ky=0; ky < kernel_size; ky++) {
        y = idy - kernel_size/2 + ky;
        if (y < 0) continue;
        
        index = x * cols + y;
        k_index = kx * kernel_size + ky;
        
        sum += (int) image[index] * kernel[k_index];
    }
    
    atomicAdd((float*)(blurred_image + target_index), sum);
  }
  """)


def get_gaussian_kernel(kernel_size):
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel = kernel.reshape(1, kernel_size)
    GF = np.dot(kernel.T, kernel)

    return GF


def gauss_gpu(input_mat, kernel_size):
    kernel = get_gaussian_kernel(kernel_size)
    kernel = np.asarray(kernel, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
    print(input_mat.shape)
    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    threadsInBlock = 32
    grid = (rows // threadsInBlock + 1, cols // threadsInBlock + 1, 1)
    block = (threadsInBlock, threadsInBlock, 1)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_gpu = gpuarray.to_gpu(kernel)
    blurred_mat = np.zeros((rows, cols), dtype=np.float32)

    gauss_gpu_kernel = mod.get_function("gauss_gpu_kernel")
    start_gpu_internal_time = time.time()
    gauss_gpu_kernel(input_mat_gpu, kernel_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(blurred_mat), block=block, grid=grid)
    print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return blurred_mat


def gauss_gpu_v2(input_mat, kernel_size):
    if(kernel_size > 255):
        print("Kernel max size exceeds")
        return

    kernel = get_gaussian_kernel(kernel_size)
    kernel = np.asarray(kernel, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
    print(input_mat.shape)
    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    threadsInBlock = int(np.sqrt(1024//kernel_size))
    grid = (rows // threadsInBlock + 1, cols // threadsInBlock + 1, 1)
    block = (threadsInBlock, threadsInBlock, kernel_size)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_gpu = gpuarray.to_gpu(kernel)
    blurred_mat = np.zeros((rows, cols), dtype=np.float32)

    gauss_gpu_kernel = mod.get_function("gauss_gpu_kernel_v2")
    start_gpu_internal_time = time.time()
    gauss_gpu_kernel(input_mat_gpu, kernel_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(blurred_mat), block=block, grid=grid)
    print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return blurred_mat


def gauss_cpu(input_mat, kernel_size):
    kernel = get_gaussian_kernel(kernel_size)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    blurred_mat = np.zeros((rows, cols), dtype=np.uint8)

    it_blurred_mat = np.nditer(blurred_mat, flags=['multi_index'])
    while not it_blurred_mat.finished:
        blurred_mat_indices = it_blurred_mat.multi_index
        blurred_mat_index_x = blurred_mat_indices[0]
        blurred_mat_index_y = blurred_mat_indices[1]

        x_start = max(blurred_mat_index_x - kernel_size//2, 0)
        x_end = min(blurred_mat_index_x + kernel_size // 2 + 1, rows)
        y_start = max(blurred_mat_index_y - kernel_size // 2, 0)
        y_end = min(blurred_mat_index_y + kernel_size // 2 + 1, cols)

        k_x_start = kernel_size - (x_end - x_start)
        k_y_start = kernel_size - (y_end - y_start)
        # print(np.asarray(input_mat[x_start:x_end, y_start:y_end]).shape)
        blurred_mat_value = np.sum(input_mat[x_start:x_end, y_start:y_end] * kernel[k_x_start:, k_y_start:])

        blurred_mat[blurred_mat_index_x][blurred_mat_index_y] = blurred_mat_value
        it_blurred_mat.iternext()

    return blurred_mat


im = cv2.imread('Two_lane_city_streets.jpg')
im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
# blurred_9 = gauss_gpu_v2(im, 9)

start_gpu_time = time.time()
blurred = gauss_gpu(im, 21)
print("time taken for gpu--- %s seconds ---" % (time.time() - start_gpu_time))

start_gpu_v2_time = time.time()
blurred_v2 = gauss_gpu_v2(im, 21)
print("time taken for gpu v2--- %s seconds ---" % (time.time() - start_gpu_v2_time))

start_cpu_time = time.time()
blurred_cpu = cv2.GaussianBlur(im, (21, 21), 0)
print("time taken for cpu--- %s seconds ---" % (time.time() - start_cpu_time))

# start_custom_cpu_time = time.time()
# blurred__custom_cpu = gauss_cpu(im, 21)
# print("time taken for custom cpu--- %s seconds ---" % (time.time() - start_custom_cpu_time))
# img = np.asarray(im, dtype=np.uint8)
# edges = cv2.Canny(im, 25, 255)

plt.subplot(311), plt.imshow(im, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(blurred, cmap='gray')
plt.subplot(313), plt.imshow(blurred_v2, cmap='gray')
plt.show()
