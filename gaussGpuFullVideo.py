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

  __global__ void gauss_gpu_full_video(unsigned char *image, float *kernel, int kernel_size, int rows, int cols, float *blurred_image) {
    int idx = threadIdx.x + (blockIdx.y * blockDim.x );
    int idy = threadIdx.y + blockIdx.z * blockDim.y;
    int frameNo = blockIdx.x;

    if(idx > rows || idy > cols) {
        return ;
    }

    int target_index = frameNo * rows * cols + idx * cols + idy;

    int x, y, index, k_index;

    float sum=0;
    int kx = threadIdx.z;
    x = idx - kernel_size/2 + kx;
    if (x < 0) return;
    for (int ky=0; ky < kernel_size; ky++) {
        y = idy - kernel_size/2 + ky;
        if (y < 0) continue;

        index = frameNo * rows * cols + x * cols + y;
        k_index = kx * kernel_size + ky;
        //printf("%d ", image[index]);
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


def gauss_gpu_full_video(input_mat, kernel_size):
    if (kernel_size > 255):
        print("Kernel max size exceeds")
        return

    kernel = get_gaussian_kernel(kernel_size)
    kernel = np.asarray(kernel, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.uint8)
    print(input_mat.shape)
    input_size = input_mat.shape
    frame_count = input_size[0]
    rows = input_size[1]
    cols = input_size[2]

    threadsInBlock = int(np.sqrt(1024 // kernel_size))
    grid = (frame_count, rows // threadsInBlock + 1, cols // threadsInBlock + 1)
    block = (threadsInBlock, threadsInBlock, kernel_size)

    input_mat_gpu = gpuarray.to_gpu(input_mat)
    kernel_gpu = gpuarray.to_gpu(kernel)
    blurred_mat = np.zeros((frame_count, rows, cols), dtype=np.float32)

    gauss_gpu_full_video = mod.get_function("gauss_gpu_full_video")
    start_gpu_internal_time = time.time()
    gauss_gpu_full_video(input_mat_gpu, kernel_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols),
                         cuda.InOut(blurred_mat), block=block, grid=grid)

    input_mat_gpu.gpudata.free()
    kernel_gpu.gpudata.free()
    #     print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return blurred_mat

video_capture = cv2.VideoCapture('SampleVideo_720x480_10mb.mp4')
fps = video_capture.get(cv2.CAP_PROP_FPS)

length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('fps: ' + str(fps))
print('length: ' + str(length))
print('width: ' + str(width))
print('height: ' + str(height))

frame_list = list()

num_frames = 1
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    num_frames = num_frames + 1
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_list.append(frame)
    if num_frames > 4:
        break

print('len: ' + str(len(frame_list)))
print(frame_list)
gauss_gpu_full_video(frame_list, 21)

# file = open("data.csv", "w")

# cnt=0
# for f in frame_list:
# #     im = cv2.imread('Two_lane_city_streets.jpg')
#     data = analyze(f)
#     file.write(', '.join(map(str, [v for k,v in data.items()])))
#     file.write('\n')

#     cnt = cnt + 1
#     print ('done %s' % cnt)

# file.close()


# analyze(frame_list[0])