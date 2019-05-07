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
    __device__ unsigned long long atomicMul(unsigned long long* address, unsigned long long val){ 
        unsigned long long int* address_as_ull = (unsigned long long int*)address; 
        unsigned long long int old = *address_as_ull, assumed; 
        do { 
        assumed = old; 
        old = atomicCAS(address_as_ull, assumed, val * assumed); 
        } while (assumed != old); return old;
    }     

  __global__ void calc_factorials_kernel(unsigned int *input_array, int input_len, int threads_in_block, unsigned long long *factorials) {
    unsigned long long input_index = blockIdx.x;
    if(input_index >= input_len){
        return;
    }
    
    int input = input_array[input_index];
    //int start_position = threadIdx.x + 1;
    int limit = input / threads_in_block;
    if (threadIdx.x < input % threads_in_block) {
        limit++;
    }
    int start_position = threadIdx.x * limit + 1;
    if (threadIdx.x >= input % threads_in_block) {
        start_position += input % threads_in_block;
    }
    
    int end = min(start_position + limit, input + 1);
    unsigned long long partial_product = 1;
    
    for(int i = start_position; i < end; i++){
        partial_product *= i;
        //printf("input index- %d %d %d %lld \\n", input_index, start_position, i, partial_product);
    }
    atomicMul((factorials + input_index), partial_product);
  }
  /*__global__ void calculate_LU_element(float *A, float *L, float *U, int i, int n){
        int k = threadIdx.y + threadIdx.x * blockDim.x;
        int row = k / n;
        int col = k % n;
        int pivotValue = A[i * n + i];
        
        if(row == 0)
            U[(row + i) * n + col + i] = A[(row + i) * n + col + i];
        elseif(col == 0)
            L[(row + i) * n + col + i] /=  pivotValue;
        else
            A[(row + i) * n + col + i] -= L[row * n + i] / pivotValue * U[i * n +col];
	}*/
  """)


def calc_factorials(input_list):
    input_len = len(input_list)

    threads_in_block = 3
    numpy_one = np.int32(1)
    grid = (input_len, 1, 1)
    block = (threads_in_block, 1, 1)

    input_array = np.asarray(input_list, dtype=np.uint32)
    input_array_gpu = gpuarray.to_gpu(input_array)
    factorials = np.ones((input_len,), dtype=np.uint64)

    calc_factorials_kernel = mod.get_function("calc_factorials_kernel")
    start_gpu_internal_time = time.time()
    calc_factorials_kernel(input_array_gpu, np.int32(input_len), np.int32(threads_in_block), cuda.InOut(factorials), block=block, grid=grid)

    input_array_gpu.gpudata.free()
    #     print("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time))

    return factorials


def lu_decomposition(a, n):
    for i in range(n):
        threads_in_block = n
        grid = (1, 1, 1)
        block = (threads_in_block, threads_in_block, 1)


print(calc_factorials([1, 5 , 7, 14, 18]))
