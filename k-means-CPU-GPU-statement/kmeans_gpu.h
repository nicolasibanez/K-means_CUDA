////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Macro for error checking
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                            fprintf(stderr, "Error in %s at %s:%d\n", __func__, __FILE__, __LINE__); \
                            exit(-1);} \
                       } while(0)
    
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)

#define CHECK_CUDA_SUCCESS(exp,msg)   {if ((exp) != cudaSuccess) {\
                                         fprintf(stderr,"Error on CUDA operation (%s)\n", msg);\
                                         exit(EXIT_FAILURE);}\
                                      }

#define CHECK_CUBLAS_SUCCESS(exp,msg)   {int r = (exp); \
                                         if (r != CUBLAS_STATUS_SUCCESS) {\
                                           fprintf(stderr,"Error (%d) on CUBLAS operation (%s)\n", r, msg);\
                                           exit(EXIT_FAILURE);}\
                                        }

// How to do error checking in CUDA
// From https://codeyarns.com/tech/2011-03-02-how-to-do-error-checking-in-cuda.html
// Define this to turn on error checking
#define CUDA_ERROR_CHECK
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err ) {
    fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
             file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  /*
  err = cudaDeviceSynchronize();
  if ( cudaSuccess != err ) {
    fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
             file, line, cudaGetErrorString( err ) );
    exit( -1 );
  }*/
#endif

  return;
}

// Functions in the module
void gpu_Init();
void gpu_Finalize();
void gpu_SetDataOnGPU();
void gpu_GetResultOnCPU();
void gpu_Kmeans();
