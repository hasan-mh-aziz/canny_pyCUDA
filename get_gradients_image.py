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
  __device__ float get_gradient_sum_row(float *image, float *kernel, int i_x, int k_x, int idy, int kernel_size, int cols, int idx){
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

  __global__ void get_gradients_gpu_kernel(float *image, float *kernel_x, float *kernel_y, int kernel_size, int rows, int cols, float *gradients_x, float *gradients_y) {
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
  
  __device__ int serialize_index(int x, int y, int rows, int cols) {
      return x*cols + y;
  }
  
  __global__ void get_gradients_gpu_kernel_v2(float *G, float *sobel_x, float *sobel_y, int kernel_size, int rows, int cols, float *gradients_x, float *gradients_y) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx >= rows-1 || idy >= cols-1) {
        return ;
    }
    
    if (idx <1 || idy <1) {
        return;
    }

    int target_index = idx * cols + idy;

    int x = idx;
    int y = idy;
    float px = (sobel_x[serialize_index(0, 0, 3, 3)] * G[serialize_index(x-1, y-1, rows, cols)]) + (sobel_x[serialize_index(0, 1, 3, 3)] * G[serialize_index(x, y-1, rows, cols)]) + \
     (sobel_x[serialize_index(0, 2, 3, 3)] * G[serialize_index(x+1, y-1, rows, cols)]) + (sobel_x[serialize_index(1, 0, 3, 3)] * G[serialize_index(x-1, y, rows, cols)]) + \
     (sobel_x[serialize_index(1, 1, 3, 3)] * G[serialize_index(x, y, rows, cols)]) + (sobel_x[serialize_index(1, 2, 3, 3)] * G[serialize_index(x+1, y, rows, cols)]) + \
     (sobel_x[serialize_index(2, 0, 3, 3)] * G[serialize_index(x-1, y+1, rows, cols)]) + (sobel_x[serialize_index(2, 1, 3, 3)] * G[serialize_index(x, y+1, rows, cols)]) + \
     (sobel_x[serialize_index(2, 2, 3, 3)] * G[serialize_index(x+1, y+1, rows, cols)]);

    float py = (sobel_y[serialize_index(0, 0, 3, 3)] * G[serialize_index(x-1, y-1, rows, cols)]) + (sobel_y[serialize_index(0, 1, 3, 3)] * G[serialize_index(x, y-1, rows, cols)]) + \
     (sobel_y[serialize_index(0, 2, 3, 3)] * G[serialize_index(x+1, y-1, rows, cols)]) + (sobel_y[serialize_index(1, 0, 3, 3)] * G[serialize_index(x-1, y, rows, cols)]) + \
     (sobel_y[serialize_index(1, 1, 3, 3)] * G[serialize_index(x, y, rows, cols)]) + (sobel_y[serialize_index(1, 2, 3, 3)] * G[serialize_index(x+1, y, rows, cols)]) + \
     (sobel_y[serialize_index(2, 0, 3, 3)] * G[serialize_index(x-1, y+1, rows, cols)]) + (sobel_y[serialize_index(2, 1, 3, 3)] * G[serialize_index(x, y+1, rows, cols)]) + \
     (sobel_y[serialize_index(2, 2, 3, 3)] * G[serialize_index(x+1, y+1, rows, cols)]);
     
     gradients_x[target_index] = px;
     gradients_y[target_index] = py;
  }
  
  __global__ void sobeldir_angles_gpu(float *image, int rows, int cols, float *sobeldir_angles) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx > rows || idy > cols) {
        return ;
    }

    int target_index = idx * cols + idy;

    if ((image[target_index]<22.5 &&  image[target_index]>=0) ||
        (image[target_index]>=157.5 &&  image[target_index]<202.5) ||
        (image[target_index]>=337.5 &&  image[target_index]<=360)) {
            sobeldir_angles[target_index]=0;
    }
    else if ((image[target_index]>=22.5 &&  image[target_index]<67.5) ||
             (image[target_index]>=202.5 &&  image[target_index]<247.5)) {
            sobeldir_angles[target_index]=45;
    }
    else if ((image[target_index]>=67.5 &&  image[target_index]<112.5) ||
             (image[target_index]>=247.5 &&  image[target_index]<292.5)) {
            sobeldir_angles[target_index]=90;
    }
    else {
            sobeldir_angles[target_index]=135;
    }
  }

  __global__ void non_max_suppression_gpu(float *sobeloutmag_mat_gpu, float *sobeloutdir_mat_gpu, int rows, int cols, float *mag_sup) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= rows-1 || idy >= cols-1) {
        return ;
    }
    
    if (idx <1 || idy <1) {
        return;
    }

    int target_index = idx * cols + idy;

    mag_sup[target_index] = sobeloutmag_mat_gpu[target_index];
        
    if (sobeloutdir_mat_gpu[target_index] == 0) 
    {
        if ((sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx, idy+1, rows, cols)]) ||
            (sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx, idy-1, rows, cols)]))
        {
            mag_sup[target_index] = 0;
        }
    }
    else if (sobeloutdir_mat_gpu[target_index] == 45)
    {
        if ((sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx-1, idy+1, rows, cols)]) ||
            (sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx+1, idy-1, rows, cols)]))
        {
            mag_sup[target_index] = 0;
        }
    }
    else if (sobeloutdir_mat_gpu[target_index] == 90)
    {
        if ((sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx+1, idy, rows, cols)]) ||
           (sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx-1, idy, rows, cols)]))
        {
            mag_sup[target_index] = 0;
        }
    }
    else
    {
        if ((sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx+1, idy+1, rows, cols)]) ||
            (sobeloutmag_mat_gpu[target_index] <= sobeloutmag_mat_gpu[serialize_index(idx-1, idy-1, rows, cols)]))
        {
            mag_sup[target_index] = 0;
        }
    }
  }
  
  __global__ void gnh_gnl_gpu(float *mag_sup, int rows, int cols, int th, int tl, float *gnh, float *gnl) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= rows || idy >= cols) {
        return ;
    }
    
    int target_index = idx * cols + idy;

    if (mag_sup[target_index]>=th)
    {
        gnh[target_index]=mag_sup[target_index];
    }
    if (mag_sup[target_index]>=tl)
    {
        gnl[target_index]=mag_sup[target_index];
    }
  }
  
  __device__ void traverse(int i, int j, int rows, int cols, float *gnh, float *gnl) {
      if (i<0 || i>=rows) return;
      if (j<0 || j>=cols) return;
      
      int x[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
      int y[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
      for (int k = 0; k < 8; k++)
      {
        if (gnh[serialize_index(i+x[k], j+y[k], rows, cols)] == 0 && gnl[serialize_index(i+x[k], j+y[k], rows, cols)] != 0)
        {
          gnh[serialize_index(i+x[k], j+y[k], rows, cols)] = 1;
          traverse(i+x[k], j+y[k], rows, cols, gnh, gnl);
        }
      }
  }
  __global__ void edge_linking_gpu(int rows, int cols, float *gnh, float *gnl) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x );
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx >= rows-1 || idy >= cols-1) {
        return ;
    }
    
    if (idx <1 || idy <1) {
        return;
    }
    
    int target_index = idx * cols + idy;

    if (gnh[target_index]>0) {
        gnh[target_index]=1;
        //traverse(idx, idy, rows, cols, gnh, gnl);
    }
  }

  """)


def get_gradient_image_gpu(input_mat, kernel_x, kernel_y, kernel_size):
    kernel_x = np.asarray(kernel_x, dtype=np.float32)
    kernel_y = np.asarray(kernel_y, dtype=np.float32)
    input_mat = np.asarray(input_mat, dtype=np.float32)

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

    get_gradients_gpu_kernel = mod.get_function("get_gradients_gpu_kernel_v2")
    start_gpu_internal_time = time.time()
#     print(kernel_x)
    get_gradients_gpu_kernel(input_mat_gpu, kernel_x_gpu, kernel_y_gpu, np.int32(kernel_size), np.int32(rows), np.int32(cols), cuda.InOut(gradients_x), cuda.InOut(gradients_y), block=block, grid=grid)
#     print("time taken for gpu internally for single row--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return gradients_x, gradients_y

def get_sobeldir_angles_gpu(sobeloutdir):
    sobeloutdir_mat = np.asarray(sobeloutdir, dtype=np.float32)
    sobeloutdir_mat_size = sobeloutdir_mat.shape
    rows = sobeloutdir_mat_size[0]
    cols = sobeloutdir_mat_size[1]
    
    threads_in_block = 32
    grid = (rows // threads_in_block + 1, cols // threads_in_block + 1, 1)
    block = (threads_in_block, threads_in_block, 1)
    
    sobeloutdir_mat_gpu = gpuarray.to_gpu(sobeloutdir_mat)
    sobeldir_angles = np.zeros((rows, cols), dtype=np.float32)

    sobeldir_angles_gpu = mod.get_function("sobeldir_angles_gpu")
    sobeldir_angles_gpu(sobeloutdir_mat_gpu, np.int32(rows), np.int32(cols), cuda.InOut(sobeldir_angles), block=block, grid=grid)

    return sobeldir_angles
    
def get_non_max_suppression_gpu(sobeloutmag, sobeloutdir):
    sobeloutmag_mat = np.asarray(sobeloutmag, dtype=np.float32)
    sobeloutdir_mat = np.asarray(sobeloutdir, dtype=np.float32)
    input_size = sobeloutmag_mat.shape
    rows = input_size[0]
    cols = input_size[1]
    
    threads_in_block = 32
    grid = (rows // threads_in_block + 1, cols // threads_in_block + 1, 1)
    block = (threads_in_block, threads_in_block, 1)
    
    sobeloutmag_mat_gpu = gpuarray.to_gpu(sobeloutmag_mat)
    sobeloutdir_mat_gpu = gpuarray.to_gpu(sobeloutdir_mat)
    mag_sup = np.zeros((rows, cols), dtype=np.float32)

    non_max_suppression_gpu = mod.get_function("non_max_suppression_gpu")
    non_max_suppression_gpu(sobeloutmag_mat_gpu, sobeloutdir_mat_gpu, np.int32(rows), np.int32(cols), cuda.InOut(mag_sup), block=block, grid=grid)

    return mag_sup

def get_gnh_gnl_gpu(map_sup, th, tl):
    map_sup_mat = np.asarray(map_sup, dtype=np.float32)
    input_size = map_sup_mat.shape
    rows = input_size[0]
    cols = input_size[1]
    
    threads_in_block = 32
    grid = (rows // threads_in_block + 1, cols // threads_in_block + 1, 1)
    block = (threads_in_block, threads_in_block, 1)
    
    map_sup_mat_gpu = gpuarray.to_gpu(map_sup_mat)
    gnh = np.zeros((rows, cols), dtype=np.float32)
    gnl = np.zeros((rows, cols), dtype=np.float32)

    gnh_gnl_gpu = mod.get_function("gnh_gnl_gpu")
    gnh_gnl_gpu(map_sup_mat_gpu, np.int32(rows), np.int32(cols), np.int32(th), np.int32(tl), cuda.InOut(gnh), cuda.InOut(gnl), block=block, grid=grid)

    return gnh, gnl

def get_edge_linking_gpu(gnh, gnl):
    gnh_mat = np.asarray(gnh, dtype=np.float32)
    gnl_mat = np.asarray(gnl, dtype=np.float32)
    input_size = gnh_mat.shape
    rows = input_size[0]
    cols = input_size[1]
    
    threads_in_block = 32
    grid = (rows // threads_in_block + 1, cols // threads_in_block + 1, 1)
    block = (threads_in_block, threads_in_block, 1)
    
    gnh_mat_gpu = np.asarray(gnh, dtype=np.float32).copy()
    gnl_mat_gpu = np.asarray(gnl, dtype=np.float32).copy()

    edge_linking_gpu = mod.get_function("edge_linking_gpu")
    edge_linking_gpu(np.int32(rows), np.int32(cols), cuda.InOut(gnh_mat_gpu), cuda.InOut(gnl_mat_gpu), block=block, grid=grid)

    return gnh_mat_gpu

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

width = img.size[1]
height = img.size[0]

imagesToPlot = []

imagesToPlot.append((imgdata.copy(), 'original gray'))

imagesToPlot.append((image.copy(), 'gaussian'))


sobelout = Image.new('L', img.size)                                       #empty image
gradients_x = numpy.array(sobelout, dtype = np.float32)
gradients_y = numpy.array(sobelout, dtype = np.float32)

gradients_x, gradients_y = get_gradient_image_gpu(image, sobel_x, sobel_y, 3)

# TODO: parallel
sobeloutmag = scipy.hypot(gradients_x, gradients_y)
sobeloutdir = scipy.arctan2(gradients_y, gradients_x)

imagesToPlot.append((sobeloutmag.copy(), 'sobeloutmag'))
imagesToPlot.append((sobeloutdir.copy(), 'sobeloutdir'))

sobeloutdir = get_sobeldir_angles_gpu(sobeloutdir)

imagesToPlot.append((sobeloutdir.copy(), 'dirquantize'))

mag_sup = get_non_max_suppression_gpu(sobeloutmag, sobeloutdir)

imagesToPlot.append((mag_sup.copy(), 'mag_sup'))

m = numpy.max(mag_sup)
th = 0.2*m
tl = 0.1*m

gnh, gnl = get_gnh_gnl_gpu(mag_sup, th, tl)

imagesToPlot.append((gnl.copy(), 'gnl_before_minus'))

gnl = gnl-gnh

imagesToPlot.append((gnl.copy(), 'gnl_after_minus'))

imagesToPlot.append((gnh.copy(), 'gnh'))

gnh = get_edge_linking_gpu(gnh, gnl)

imagesToPlot.append((gnh.copy(), 'final'))

plt.rcParams['figure.figsize'] = [20, 10]
for index, d in enumerate(imagesToPlot, start=1):
    plt.subplot(2, 5, index)
    # plt.imshow(cv2.cvtColor(d[0].astype(np.uint8), cv2.COLOR_GRAY2RGB))
    plt.imshow(d[0].astype(np.uint8), cmap='gray')
    plt.title(d[1])

plt.show()
