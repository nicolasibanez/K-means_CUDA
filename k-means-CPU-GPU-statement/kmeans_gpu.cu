////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>              // for CUBLAS_GEAM
#include <curand_kernel.h>          // for CURAND_UNIFORM
#include <float.h>

#include "main.h"
#include "init.h"
#include "kmeans_gpu.h"


/*----------------------------------------------------------------------------*/
/* Define pointers on GPU variables, and GPU symbols                          */
/*----------------------------------------------------------------------------*/
// Choose the ONE you need (big array)
T_real *GPU_instance;      //[NB_INSTANCES][NB_DIMS] --> [NB_INSTANCES * NB_DIMS]
T_real *GPU_instance_T;    //[NB_DIMS][NB_INSTANCES] --> [NB_DIMS * NB_INSTANCES]

// Choose the one you need or both (small arrays)
T_real *GPU_centroid;      //[NB_CLUSTERS][NB_DIMS] --> [NB_CLUSTERS * NB_DIMS]
T_real *GPU_centroid_T;    //[NB_DIMS][NB_CLUSTERS] --> [NB_DIMS * NB_CLUSTERS]

int *GPU_label;             //[NB_INSTANCES] Label of each point
int *GPU_change;            //[NB_INSTANCES] Flag recording the change of label
int *GPU_count;             //[NB_CLUSTERS]  Count of instance points in each cluster
int *GPU_failed;

__device__ unsigned long long GPU_change_total; // nb of label changes at current iter
unsigned long long *AdrGPU_change_total = NULL;

curandState *devStates;     // To use curand
cublasHandle_t cublasHandle;// To activate cublas (speedup the GPU!)


/*-------------------------------------------------------------------------------*/
/* Init and finalize the GPU device.                                             */
/*-------------------------------------------------------------------------------*/
void gpu_Init()
{
  cuInit(0);

  // Allocate memory space for GPU arrays
  
  // Choose ONE (Big array)
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_instance, sizeof(T_real)*NB_INSTANCES*NB_DIMS), "Dynamic allocation for GPU_instance");
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_instance_T, sizeof(T_real)*NB_DIMS*NB_INSTANCES), "Dynamic allocation for GPU_instance_T");
  

  // Choose one or both (small array)
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroid, sizeof(T_real)*NB_CLUSTERS*NB_DIMS), "Dynamic allocation for GPU_centroid");
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroid_T, sizeof(T_real)*NB_DIMS*NB_CLUSTERS), "Dynamic allocation for GPU_centroid_T");
  
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_label, sizeof(int)*NB_INSTANCES), "Dynamic allocation for GPU_label");
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_change, sizeof(int)*NB_INSTANCES), "Dynamic allocation for GPU_change");
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_count, sizeof(int)*NB_CLUSTERS), "Dynamic allocation for GPU_count");

  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_failed, sizeof(int)*1), "Dynamic allocation for GPU_failed");
 
  // Initialize an array of "curandState) for using curand
  CHECK_CUDA_SUCCESS(cudaMalloc((void**) &devStates, sizeof(curandState)*NB_CLUSTERS), "Dynamic allocation for devStates");

  // Get address of GPU symbols
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **) &AdrGPU_change_total, GPU_change_total), 
                     "Get the address of GPU_change_total");
                     
  // Initialize CUBLAS lib usage
  CHECK_CUBLAS_SUCCESS(cublasCreate(&cublasHandle), "Init of the CUBLAS lib handle"); 
}

void gpu_Finalize()
{
  // Free dynamic allocations (function of the arrays you used)
  CHECK_CUDA_SUCCESS(cudaFree(GPU_instance), "Free the dynamic allocation for GPU_instance");
  CHECK_CUDA_SUCCESS(cudaFree(GPU_instance_T), "Free the dynamic allocation for GPU_instance_T");

  CHECK_CUDA_SUCCESS(cudaFree(GPU_centroid), "Free the dynamic allocation for GPU_centroid");
  CHECK_CUDA_SUCCESS(cudaFree(GPU_centroid_T), "Free the dynamic allocation for GPU_centroid_T");
  CHECK_CUDA_SUCCESS(cudaFree(GPU_label), "Free the dynamic allocation for GPU_label");
  CHECK_CUDA_SUCCESS(cudaFree(GPU_change), "Free the dynamic allocation for GPU_change");
  CHECK_CUDA_SUCCESS(cudaFree(GPU_count), "Free the dynamic allocation for GPU_count");

  CHECK_CUDA_SUCCESS(cudaFree(GPU_failed), "Free the dynamic allocation for GPU_failed");
  
  // Free array of curandStates
  CHECK_CUDA_SUCCESS(cudaFree(devStates), "Free the dynamic allocation for devStates");

  // Free CUBLAS lib usage
  CHECK_CUBLAS_SUCCESS(cublasDestroy(cublasHandle), "Free the CUBLAS lib");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                   */
/*-------------------------------------------------------------------------------*/
void gpu_SetDataOnGPU()
{
  // Transfer instance[] or instance_T[] .... as you want                  // TO DO
  CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_instance, instance,
                                sizeof(T_real)*NB_DIMS*NB_INSTANCES, 
                                cudaMemcpyHostToDevice),
                      "Transfer instance...");


  
  T_real alpha = 1.0f;
  T_real beta = 0.0f;
  CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  NB_INSTANCES, NB_DIMS,
                                  &alpha, GPU_instance, NB_DIMS,
                                  &beta, NULL, NB_INSTANCES,
                                  GPU_instance_T, NB_INSTANCES), 
                      "Use CUBLAS_GEAM to transpose GPU_instance");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                        */
/*-------------------------------------------------------------------------------*/
void gpu_GetResultOnCPU()
{
  // Transfer labels computed on GPU, to the CPU                           // TO DO
  CHECK_CUDA_SUCCESS(cudaMemcpy(label, GPU_label, 
                               sizeof(int)*NB_INSTANCES, 
                               cudaMemcpyDeviceToHost),
                    "Transfer labels...");

  // Transfer final centroids computed on GPU, to the CPU                  // TO DO
  // MIGHT NEED TO TRANSPOSE !

  T_real alpha = 1.0f;
  T_real beta = 0.0f;
  CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  NB_DIMS, NB_CLUSTERS,
                                  &alpha, GPU_centroid_T, NB_CLUSTERS,
                                  &beta, NULL, NB_DIMS,
                                  GPU_centroid, NB_DIMS), 
                      "Use CUBLAS_GEAM to transpose GPU_centroid_T");

  CHECK_CUDA_SUCCESS(cudaMemcpy(centroid, GPU_centroid,
                               sizeof(T_real)*NB_CLUSTERS*NB_DIMS,
                               cudaMemcpyDeviceToHost),
                    "Transfer centroids...");
}


/*-------------------------------------------------------------------------------*/
/* Initialize the random generator used for each centroid                        */
/*-------------------------------------------------------------------------------*/
__global__ void kernel_SetupcuRand(curandState *state)
{
  int centroidIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (centroidIdx < NB_CLUSTERS) {
    curand_init(4321, centroidIdx, 0, &state[centroidIdx]);
  }
}


/*-------------------------------------------------------------------------------*/
/* Select the initial centroids (uniformly at random) from the input data        */
/*-------------------------------------------------------------------------------*/
__global__ void kernel_InitializeCentroids(curandState *state, T_real *GPU_centroid_T, T_real *GPU_instance_T)
                                           /*T_real *GPU_centroid OR *GPU_centroid_T*/ 
                                           /*T_real *GPU_instance OR *GPU_instance_T*/
{
  int centroidIdx = threadIdx.x + blockIdx.x * blockDim.x;

  // Each initial centroid will be one of the input data 
  if (centroidIdx < NB_CLUSTERS) {
  
    // Get the current state of the random generator of the centroid
    curandState localState = state[centroidIdx];
    
    // Compute an idx value in [0, NB_INSTANCES - 1]: select an input data
    // Note: curand_uniform() returns a pseudo-random float in the range [0.0, 1.0[
    int idx = floor(NB_INSTANCES * CURAND_UNIFORM(&localState));
    
    // Set the centroid coordinates with the selected input data coordinates 
    for (int j = 0; j < NB_DIMS; j++)                                   // TO DO
    //  GPU_centroid[...] = GPU_instance[...]
    //  or GPU_centroid_T[...] = GPU_instance_T[...]
    //  or GPU_centroid[...] = GPU_instance_T[...]
    //  or GPU_centroid_T[...] = GPU_instance[...]

      // SHARED MEMORY TO IMPROVE? (should use Tranpose matrix of GPU_intance instead)
      // GPU_centroid_T[j][centroidIdx] = GPU_instance[idx][j];
      // GPU_centroid_T[j * NB_CLUSTERS + centroidIdx] = GPU_instance[idx * NB_DIMS + j];
      GPU_centroid_T[j * NB_CLUSTERS + centroidIdx] = GPU_instance_T[j * NB_INSTANCES + idx];

  }
}


/*-------------------------------------------------------------------------------*/
/* Compute distances and Assign each point to its nearest centorid               */
/*-------------------------------------------------------------------------------*/
__global__ void kernel_ComputeAssign(T_real *GPU_instance_T, T_real *GPU_centroid_T, int *GPU_label, unsigned long long *AdrGPU_change_total)
{
  // TO DO

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int closest_centroid_idx = 0;
  T_real min_dist = REAL_MAX;

  if (idx < NB_INSTANCES) {
    for (int i = 0; i < NB_CLUSTERS; ++i) {
      T_real distance = 0.0;
      for (int j = 0; j < NB_DIMS; ++j) {
        T_real temp = (GPU_instance_T[j * NB_INSTANCES + idx] - GPU_centroid_T[j * NB_CLUSTERS + i]);
        distance += temp*temp;
      }
      if (distance < min_dist) {
        min_dist = distance;
        closest_centroid_idx = i;
      }
    }
    if (GPU_label[idx] != closest_centroid_idx) {
      atomicAdd(AdrGPU_change_total, 1);
      GPU_label[idx] = closest_centroid_idx;
    }
  }
 
  // Note:
  //   The "atomic add" on global GPU var could be useful:
  //     atomicAdd(Adr_of_GPU_var, Integer_Value_to_Add);
  //   Warning: time consumming function
}


__global__ void OLD_kernel_UpdateCentroid_Step1(T_real *GPU_instance_T, T_real *GPU_centroid_T, int *GPU_label, int *GPU_count)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ T_real sh_instance[BLOCK_SIZE_X_N];
  __shared__ T_real sh_count[BLOCK_SIZE_X_N];
  

  if (idx < NB_INSTANCES) {
    for(int clusterIdx = 0; clusterIdx < NB_CLUSTERS; ++clusterIdx){
      for(int dim = 0; dim < NB_DIMS; ++dim) {
        if (GPU_label[idx] == clusterIdx) {
          sh_instance[threadIdx.x] = GPU_instance_T[dim * NB_INSTANCES + idx];
          sh_count[threadIdx.x] = 1;
        }
        else { // Divergence to test
          sh_instance[threadIdx.x] = 0;
          sh_count[threadIdx.x] = 0;
        }
        
        __syncthreads();

        // reduction : 
        // 1st step : half the threads should work
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
          if (threadIdx.x < s) {
            sh_instance[threadIdx.x] += sh_instance[threadIdx.x + s];

            // Better way?
            if(dim == 0) {
              sh_count[threadIdx.x] += sh_count[threadIdx.x + s];
            }

          }
          __syncthreads();
        }

        // atomic add in GPU_centroid_T and GPU_count
        if (threadIdx.x == 0) {
          atomicAdd(&GPU_centroid_T[dim * NB_CLUSTERS + clusterIdx], sh_instance[0]);
          if (dim == 0) {
            atomicAdd(&GPU_count[clusterIdx], sh_count[0]);
          }
        }
      } 
    }
  }
}

__global__ void OLD2_kernel_UpdateCentroid_Step1(T_real *GPU_instance_T, T_real *GPU_centroid_T, int *GPU_label, int *GPU_count)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int dim = blockIdx.y;

  __shared__ T_real sh_instance[BLOCK_SIZE_X_N];
  __shared__ T_real sh_count[BLOCK_SIZE_X_N];

  if (idx < NB_INSTANCES) {
    for(int clusterIdx = 0; clusterIdx < NB_CLUSTERS; ++clusterIdx){
      // ?
      if (GPU_label[idx] == clusterIdx) {
        sh_count[threadIdx.x] = 1;
        sh_instance[threadIdx.x] = GPU_instance_T[NB_INSTANCES*dim + idx];
      }
      else {
        sh_count[threadIdx.x] = 0;
        sh_instance[threadIdx.x] = 0;
      }

      __syncthreads();

      // reduction : 
      // 1st step : half the threads should work
      for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
          sh_instance[threadIdx.x] += sh_instance[threadIdx.x + s];
          sh_count[threadIdx.x] += sh_count[threadIdx.x + s];
        }
        __syncthreads();
      }

        // atomic add in GPU_centroid_T and GPU_count
      if (threadIdx.x == 0) {
        atomicAdd(&GPU_centroid_T[dim * NB_CLUSTERS + clusterIdx], sh_instance[0]);
        if (dim == 0) {
          atomicAdd(&GPU_count[clusterIdx], sh_count[0]);
        }
      }
    }
  }
}

__global__ void OLD3_kernel_UpdateCentroid_Step1(T_real *GPU_instance_T, T_real *GPU_centroid_T, int *GPU_label, int *GPU_count)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int dim = blockIdx.y;
  int clusterIdx = blockIdx.z;

  // if (threadIdx.x == 0) {
  //   printf("idx = %d, dim = %d\n", idx, dim);
  // }

  __shared__ T_real sh_instance[BLOCK_SIZE_X_N];
  __shared__ T_real sh_count[BLOCK_SIZE_X_N];

  if (idx < NB_INSTANCES) {
    if (GPU_label[idx] == clusterIdx) {
      sh_count[threadIdx.x] = 1;
      sh_instance[threadIdx.x] = GPU_instance_T[NB_INSTANCES*dim + idx];
    }
    else {
      sh_count[threadIdx.x] = 0;
      sh_instance[threadIdx.x] = 0;
    }


    // reduction : 
    // 1st step : half the threads should work
    // for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    //   if (threadIdx.x < s) {
    //     sh_instance[threadIdx.x] += sh_instance[threadIdx.x + s];
    //     sh_count[threadIdx.x] += sh_count[threadIdx.x + s];
    //   }
    //   __syncthreads();
    // }

    // loop unroll version :
    #if BLOCK_SIZE_X_N > 512
    __syncthreads();
    if (threadIdx.x < 512) {
      sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 512];
      sh_count[threadIdx.x] += sh_count[threadIdx.x + 512];
    }
    else
    {
      return;
    }
    #endif
    
   

    #if BLOCK_SIZE_X_N > 256
    __syncthreads();
    if (threadIdx.x < 256) {
      sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 256];
      sh_count[threadIdx.x] += sh_count[threadIdx.x + 256];
    }
    else {
      return;
    }
    #endif

    #if BLOCK_SIZE_X_N > 128
    __syncthreads();
    if (threadIdx.x < 128) {
      sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 128];
      sh_count[threadIdx.x] += sh_count[threadIdx.x + 128];
    }
    else {
      return;
    }
    #endif

    #if BLOCK_SIZE_X_N > 64
    __syncthreads();
    if (threadIdx.x < 64) {
      sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 64];
      sh_count[threadIdx.x] += sh_count[threadIdx.x + 64];
    }
    else {
      return;
    }
    #endif

    #if BLOCK_SIZE_X_N > 32
    __syncthreads();
    if (threadIdx.x < 32) {
      sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 32];
      sh_count[threadIdx.x] += sh_count[threadIdx.x + 32];
    }
    else {
      return;
    }
    #endif

    #if BLOCK_SIZE_X_N > 16
    sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 16];
    sh_count[threadIdx.x] += sh_count[threadIdx.x + 16];
    #endif

    #if BLOCK_SIZE_X_N > 8
    sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 8];
    sh_count[threadIdx.x] += sh_count[threadIdx.x + 8];
    #endif

    #if BLOCK_SIZE_X_N > 4
    sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 4];
    sh_count[threadIdx.x] += sh_count[threadIdx.x + 4];
    #endif

    #if BLOCK_SIZE_X_N > 2
    sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 2];
    sh_count[threadIdx.x] += sh_count[threadIdx.x + 2];
    #endif

    #if BLOCK_SIZE_X_N > 1
    sh_instance[threadIdx.x] += sh_instance[threadIdx.x + 1];
    sh_count[threadIdx.x] += sh_count[threadIdx.x + 1];
    #endif

    // atomic add in GPU_centroid_T and GPU_count
    if (threadIdx.x == 0) {
      atomicAdd(&GPU_centroid_T[dim * NB_CLUSTERS + clusterIdx], sh_instance[0]);
      if (dim == 0) {
        atomicAdd(&GPU_count[clusterIdx], sh_count[0]);
      }
    }
  }
}

// Version 3 : no cluster loop
__global__ void kernel_UpdateCentroid_Step1(T_real *GPU_instance_T, T_real *GPU_centroid_T, int *GPU_label, int *GPU_count)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int dim = blockIdx.y;
  int clusterIdx = blockIdx.z;

  // if (threadIdx.x == 0) {
  //   printf("idx = %d, dim = %d\n", idx, dim);
  // }

  __shared__ T_real sh_instance[BLOCK_SIZE_X_N];
  __shared__ T_real sh_count[BLOCK_SIZE_X_N];

  if (idx < NB_INSTANCES) {
    if (GPU_label[idx] == clusterIdx) {
      sh_count[threadIdx.x] = 1;
      sh_instance[threadIdx.x] = GPU_instance_T[NB_INSTANCES*dim + idx];
    }
    else {
      sh_count[threadIdx.x] = 0;
      sh_instance[threadIdx.x] = 0;
    }

    __syncthreads();

    // reduction : 
    // 1st step : half the threads should work
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sh_instance[threadIdx.x] += sh_instance[threadIdx.x + s];
        sh_count[threadIdx.x] += sh_count[threadIdx.x + s];
      }
      __syncthreads();
    }

      // atomic add in GPU_centroid_T and GPU_count
    if (threadIdx.x == 0) {
      atomicAdd(&GPU_centroid_T[dim * NB_CLUSTERS + clusterIdx], sh_instance[0]);
      if (dim == 0) {
        atomicAdd(&GPU_count[clusterIdx], sh_count[0]);
      }
    }
  }
}

/*-------------------------------------------------------------------------------*/
/* Update centroids - step 2                                                     */
/*-------------------------------------------------------------------------------*/
__global__ void kernel_UpdateCentroid_Step2(T_real *GPU_centroid_T, int *GPU_count, T_real *GPU_instance_T, int *GPU_failed)
{

  int clusterIdx = threadIdx.x + blockIdx.x * blockDim.x;
  bool failed = false;
  
  if (clusterIdx < NB_CLUSTERS) {   
    int count = GPU_count[clusterIdx];
    if (count > 0) {
      for (int dim = 0; dim < NB_DIMS; ++dim) { 
        GPU_centroid_T[dim * NB_CLUSTERS + clusterIdx] /= count;   
      }
    }
    else {
      failed = true;
    }

    if (failed){
      atomicAdd(GPU_failed, 1);
    }
  }
 // TO DO

 // Note:
 //   The "atomic add" on global GPU var could be useful:
 //     atomicAdd(Adr_of_GPU_var, Integer_Value_to_Add);
 //   Warning: time consumming function
}


/*-------------------------------------------------------------------------------*/
/* Complete clustering on GPU, with loop control on CPU                          */
/*-------------------------------------------------------------------------------*/
void gpu_Kmeans()
{
  // End criteria variables
  double tolerance = 0.0;          // tolerance will be: nb_changes / NB_INSTANCES
  int nb_iter_kmeans = 0;

  dim3 Dg, Db;
  
  // Reset the array of labels (result of the clustering) to 0
  CHECK_CUDA_SUCCESS(cudaMemset(GPU_label, 0, sizeof(int)*NB_INSTANCES), 
                     "Reset GPU_label to zeros");

  // Initialize the random generator used for each centroid
  Db.x = BLOCK_SIZE_X_C;
  Db.y = 1;
  Db.z = 1;
  Dg.x = NB_CLUSTERS/Db.x + (NB_CLUSTERS%Db.x > 0 ? 1 : 0);
  Dg.y = 1;
  Dg.z = 1;
  kernel_SetupcuRand<<<Dg,Db>>>(devStates);
  
  // Select initial centroids at random                              // TO DO
  // CudaCheckError();
  kernel_InitializeCentroids<<<Dg,Db>>>(devStates, GPU_centroid_T, GPU_instance_T);

  //CudaCheckError();

  // Note: IF NEEDED you can transpose a 2D array using CUBLAS_GEAM() function
  // Ex: Transpose GPU_centroid_T to GPU_centroid
  //
  //T_real alpha = 1.0f;
  //T_real beta = 0.0f;
  //CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle,
  //                                 CUBLAS_OP_T, CUBLAS_OP_N,
  //                                 NB_DIMS, NB_CLUSTERS,
  //                                 &alpha, GPU_centroid_T, NB_CLUSTERS,
  //                                 &beta, NULL, NB_DIMS,
  //                                 GPU_centroid, NB_DIMS), 
  //                     "Use CUBLAS_GEAM to transpose GPU_centroid_T");

  // Clustering iterative loop --------------------------------------------
  do {
    // - Reset the GPU counter of label changes at the current iteration
    CHECK_CUDA_SUCCESS(cudaMemset(AdrGPU_change_total, 0, 
                                  sizeof(unsigned long long int)*1), 
                       "Reset GPU_change_total to zero");

    // - Compute distance & Assign points to clusters 
    Db.x = BLOCK_SIZE_X_N;
    Db.y = 1;
    Db.z = 1;
    Dg.x = NB_INSTANCES/Db.x + (NB_INSTANCES%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    Dg.z = 1;
    kernel_ComputeAssign<<<Dg,Db>>>(GPU_instance_T, GPU_centroid_T, GPU_label, AdrGPU_change_total);
    CudaCheckError();

    CHECK_CUDA_SUCCESS(cudaMemcpy(&nb_changes, AdrGPU_change_total, 
                                  sizeof(unsigned long long int)*1, 
                                  cudaMemcpyDeviceToHost),
                       "Transfer GPU_change_total-->nb_changes");


    // - Update Centroids - step 1
    // -- reset the array of counters of points associated to each cluster
    //CHECK_CUDA_SUCCESS(cudaMemset(GPU_count, 0,...), "Reset GPU_count to zeros");
    // -- reset the array of centroid coordinates 
    //CHECK_CUDA_SUCCESS(cudaMemset(..., ..., ...), "Reset GPU centroids");
    
    // -- compute the number of points associated to each cluster
    //   and compute the sum of their coordinates (to compute their barycenter in next kernel)
    //   Note : you can use atomicAdd(...) ... and shared memory to reduce the nb of atomicAdd....
    // Db.x = BLOCK_SIZE_X_N;
    // Db.y = 1;
    // Db.z = 1;
    // Dg.x = NB_INSTANCES/Db.x + (NB_INSTANCES%Db.x > 0 ? 1 : 0);
    // Dg.y = 1;
    // Dg.z = 1;


    // Db.x = BLOCK_SIZE_X_N;
    // Db.y = 1;
    // Db.z = 1;
    // Dg.x = NB_INSTANCES/Db.x + (NB_INSTANCES%Db.x > 0 ? 1 : 0);
    // Dg.y = NB_DIMS; // Not in Db.y to have different shared mem for different dim
    // Dg.z = 1;

    // printf("Before grille setup\n");

    Db.x = BLOCK_SIZE_X_N;
    
    Db.y = 1;
    Db.z = 1;
    Dg.x = NB_INSTANCES/Db.x + (NB_INSTANCES%Db.x > 0 ? 1 : 0);
    // printf("Before grille setup\n");
    Dg.y = NB_DIMS; 
    // printf("Before grille setup\n");
    // printf("NB_CLUSTERS = %d\n", NB_CLUSTERS);
    Dg.z = 4;
    // printf("hardcoded\n");
    Dg.z = NB_CLUSTERS;
    // printf("Before comm\n");
    
    // printf("NB_DIMS = %d, NB_INSTANCES = %d\n", NB_DIMS, NB_INSTANCES);
    // printf("Db.x = %d, Db.y = %d, Db.z = %d\n", Db.x, Db.y, Db.z);

    // printf("Dg.z");

    // TO verify
    // intialize GPU_centroid_T and GPU_count to 0
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_centroid_T, 0, sizeof(T_real)*NB_DIMS*NB_CLUSTERS), "Reset GPU_centroid_T to zeros");
    CHECK_CUDA_SUCCESS(cudaMemset(GPU_count, 0, sizeof(int)*NB_CLUSTERS), "Reset GPU_count to zeros");
    
    // printf("Im here2");

    kernel_UpdateCentroid_Step1<<<Dg,Db>>>(GPU_instance_T, GPU_centroid_T, GPU_label, GPU_count);



    // Db.x = BLOCK_SIZE_X_C;
    // Db.y = 1;
    // Db.z = 1;
    // Dg.x = NB_CLUSTERS/Db.x + (NB_CLUSTERS%Db.x > 0 ? 1 : 0);
    // Dg.y = 1;
    // Dg.z = 1;
    // kernel_UpdateCentroid_Step2<<<Dg,Db>>>(GPU_centroid_T, GPU_count, GPU_instance_T);

    // Initialize the random generator used for each centroid
    
    

    Db.x = BLOCK_SIZE_X_C;
    Db.y = 1;
    Db.z = 1;
    Dg.x = NB_CLUSTERS/Db.x + (NB_CLUSTERS%Db.x > 0 ? 1 : 0);
    Dg.y = 1;
    Dg.z = 1;

    CHECK_CUDA_SUCCESS(cudaMemset(GPU_failed, 0, sizeof(int)*1), "Reset GPU_failed to zeros");

    kernel_UpdateCentroid_Step2<<<Dg,Db>>>(GPU_centroid_T, GPU_count, GPU_instance_T, GPU_failed);
    
    int failed = 0;

    CHECK_CUDA_SUCCESS(cudaMemcpy(&failed, GPU_failed, 
      sizeof(int)*1, 
      cudaMemcpyDeviceToHost),
      "Transfer labels 'failed'...");

    if (failed>0) {
      CHECK_CUDA_SUCCESS(cudaMemset(GPU_label, 0, sizeof(int)*NB_INSTANCES), 
                     "Reset GPU_label to zeros");


      kernel_SetupcuRand<<<Dg,Db>>>(devStates);
      
      // Select initial centroids at random                              // TO DO
      // CudaCheckError();
      kernel_InitializeCentroids<<<Dg,Db>>>(devStates, GPU_centroid_T, GPU_instance_T);
    }

    CudaCheckError();

    // - End criteria computation
    tolerance = ((double)nb_changes) / NB_INSTANCES;     
    printf("Track = %llu  Tolerance = %lf\n", nb_changes, tolerance); 
    nb_iter_kmeans++;
    
  } while (tolerance > TOL_KMEANS && nb_iter_kmeans < MAX_ITER_KMEANS);

  // To measure correct time in main.cc
  //cudaDeviceSynchronize();   // not necessary if you call CudaCheckError() 
                               // that already wait the end of last GPU op.
}
