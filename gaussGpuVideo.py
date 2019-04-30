import cv2
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

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
  
  __global__ void gauss_gpu_kernel_row(unsigned char *image, float *kernel, int kernel_size, int rows, int cols, float *blurred_image) {
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
  
  __global__ void gauss_gpu_kernel_full(unsigned char *image, float *kernel, int kernel_size, int rows, int cols, float *blurred_image) {
    int idx = blockIdx.x;
    int idy = blockIdx.y;
    
    if(idx > rows || idy > cols) {
        return ;
    }
    
    int target_index = idx * cols + idy;
    
    int x, y, index, k_index;
    
    float sum=0;
    int kx = threadIdx.x / kernel_size;
    int ky = threadIdx.x % kernel_size;
    x = idx - kernel_size/2 + kx;
    y = idy - kernel_size/2 + ky;
    
    if (x < 0 || y < 0 || kx > kernel_size || ky > kernel_size) return;
    index = x * cols + y;
    k_index = kx * kernel_size + ky;
    
    sum += (int) image[index] * kernel[k_index];
    /*for (int ky=0; ky < kernel_size; ky++) {
        y = idy - kernel_size/2 + ky;
        if (y < 0) continue;
        
        index = x * cols + y;
        k_index = kx * kernel_size + ky;
        
        sum += (int) image[index] * kernel[k_index];
    }*/
    
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
#     print(input_mat.shape)
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
#     print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return blurred_mat


def gauss_gpu_kernel_row(input_mat, kernel_size):
    if(kernel_size > 255):
        print("Kernel max size exceeds")
        return

    kernel = get_gaussian_kernel(kernel_size)
    kernel = np.asarray(kernel, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
#     print(input_mat.shape)
    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    threadsInBlock = int(np.sqrt(1024//kernel_size))
    grid = (rows // threadsInBlock + 1, cols // threadsInBlock + 1, 1)
    block = (threadsInBlock, threadsInBlock, kernel_size)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_gpu = gpuarray.to_gpu(kernel)
    blurred_mat = np.zeros((rows, cols), dtype=np.float32)

    gauss_gpu_kernel_row = mod.get_function("gauss_gpu_kernel_row")
    start_gpu_internal_time = time.time()
    gauss_gpu_kernel_row(input_mat_gpu, kernel_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(blurred_mat), block=block, grid=grid)
#     print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return blurred_mat

def gauss_gpu_kernel_full(input_mat, kernel_size):
    if(kernel_size > 255):
        print("Kernel max size exceeds")
        return

    kernel = get_gaussian_kernel(kernel_size)
    kernel = np.asarray(kernel, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
#     print(input_mat.shape)
    input_size = input_mat.shape
    rows = input_size[0]
    cols = input_size[1]

    threadsInBlock = int(np.sqrt(1024//(kernel_size * kernel_size)))
    grid = (rows // threadsInBlock + 1, cols // threadsInBlock + 1, 1)
    block = (kernel_size * kernel_size, threadsInBlock, threadsInBlock)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_gpu = gpuarray.to_gpu(kernel)
    blurred_mat = np.zeros((rows, cols), dtype=np.float32)

    gauss_gpu_kernel_full = mod.get_function("gauss_gpu_kernel_full")
    start_gpu_internal_time = time.time()
    gauss_gpu_kernel_full(input_mat_gpu, kernel_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(blurred_mat), block=block, grid=grid)
#     print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

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

def analyze(f):
    frame = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    # blurred_9 = gauss_gpu_v2(im, 9)
    data = dict()
    kernel_size = 21
    start_gpu_time = time.time()
    blurred = gauss_gpu(frame, kernel_size)
    data['gpu'] = (time.time() - start_gpu_time)
#     print("time taken for gpu kernel single --- %s seconds ---" % data['gpu'])

    start_gpu_row_time = time.time()
    blurred_kernel_row = gauss_gpu_kernel_row(frame, kernel_size)
    data['gpu_row'] = (time.time() - start_gpu_row_time)
#     print("time taken for gpu kernel row --- %s seconds ---" % data['gpu_row'])

    start_gpu_full_time = time.time()
    blurred_kernel_full = gauss_gpu_kernel_full(frame, kernel_size)
#     print("time taken for gpu kernel full --- %s seconds ---" % (time.time() - start_gpu_full_time))

    start_cpu_time = time.time()
    blurred_cpu = gauss_cpu(frame, kernel_size)
    data['gpu_single'] = (time.time() - start_cpu_time)
#     print("time taken for cpu--- %s seconds ---" % data['gpu_single'])

    return data
    # start_custom_cpu_time = time.time()
    # blurred__custom_cpu = gauss_cpu(im, 21)
    # print("time taken for custom cpu--- %s seconds ---" % (time.time() - start_custom_cpu_time))
    # img = np.asarray(im, dtype=np.uint8)
    # edges = cv2.Canny(im, 25, 255)

#     plt.subplot(311), plt.imshow(frame, cmap='gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(312), plt.imshow(blurred, cmap='gray')
#     plt.show()

video_capture = cv2.VideoCapture('SampleVideo_720x480_10mb.mp4')
fps = video_capture.get(cv2.CAP_PROP_FPS)

length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print ('fps: ' + str(fps))
print ('length: ' + str(length))
print ('width: ' + str(width))
print ('height: ' + str(height))

frame_list = list()

num_frames = 1
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret :
        break;
    num_frames = num_frames + 1
    frame_list.append(frame)
    if num_frames > 500:
        break;

print ('len: ' + str(len(frame_list)))

file = open("data.csv", "w")

cnt=0
for f in frame_list:
#     im = cv2.imread('Two_lane_city_streets.jpg')
    data = analyze(f)
    file.write(', '.join(map(str, [v for k,v in data.items()])))
    file.write('\n')
    
    cnt = cnt + 1
    print ('done %s frames' % cnt)
    
file.close()
    

