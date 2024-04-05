////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "main.h"
#include "init.h"
#include "kmeans_gpu.h"
#include "kmeans_cpu.h"


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/
int NbThreads = DEFAULT_NB_THREADS;
int OnGPUFlag = DEFAULT_ONGPUFLAG;

const char *FileDirectory = NULL;

// instance or instance_T ? choose the one you need on CPU or on GPU....
T_real *instance = NULL;      // instance[NB_INSTANCES][NB_DIMS]
T_real *instance_T = NULL;    // instance_T[NB_DIMS][NB_INSTANCES]

int    *label = NULL;         // label[NB_INSTANCES]

T_real *centroid;             // centroid[NB_CLUSTERS][NB_DIMS]
unsigned long long int *count;// count[NB_CLUSTERS]

unsigned long long int nb_changes = 0; // Nb of pts changing of label


/*-------------------------------------------------------------------------------*/
/* Toplevel function.                                                            */
/*-------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
   double tcpu1, tcpu2;
   cudaEvent_t egpu1, egpu2, egpu3, egpu4;
   float deltagpu;
   double dcalc, dcomm, dtot;
   double Vcalc, Vtot, BWcomm;

   CommandLineParsing(argc,argv);                /* Cmd line parsing.           */
   omp_set_num_threads(NbThreads);               /* Max nb of threads/node.     */

   /* K-means clustering ------------------------------------------------------ */
   fprintf(stdout,"K-means clustering %s:\n", (OnGPUFlag ? "on GPU" : "on CPU"));
   fprintf(stdout,"  - Dataset %s: %d instances in %d dimensions\n\n",
           DATA_FILE_NAME, NB_INSTANCES, NB_DIMS);

   InputData();

   dcalc = dcomm = dtot = 0.0;
   if (OnGPUFlag) {
     gpu_Init();
     
     cudaEventCreate(&egpu1);       // Create events for time measurements
     cudaEventCreate(&egpu2);
     cudaEventCreate(&egpu3);
     cudaEventCreate(&egpu4);

     cudaEventRecord(egpu1,0);
     gpu_SetDataOnGPU();            // Transfer data to the GPU
     cudaEventRecord(egpu2,0);
     gpu_Kmeans();                  // Run k-means on GPU
     cudaEventRecord(egpu3,0);
     gpu_GetResultOnCPU();          // Transfer results on CPU
     cudaEventRecord(egpu4,0);
     cudaDeviceSynchronize();       // Synchro for time measurement
    
     cudaEventElapsedTime(&deltagpu,egpu1,egpu2); 
     dcomm += deltagpu;             // GPU communication time
     cudaEventElapsedTime(&deltagpu,egpu3,egpu4);
     dcomm += deltagpu;             // GPU communication time
     cudaEventElapsedTime(&deltagpu,egpu1,egpu4);
     dtot  += deltagpu;             // GPU total time
     dcalc = dtot - dcomm;          // GPU calcul time deduction
    
     dcomm /= 1000.0;               // Convert elapsed time from ms to s
     dcalc /= 1000.0;
     dtot /= 1000.0;

     cudaEventDestroy(egpu1);       // Delete events used for time measurement
     cudaEventDestroy(egpu2);
     cudaEventDestroy(egpu3); 
     cudaEventDestroy(egpu4);

     gpu_Finalize();
     
     // Speed computations
     Vcalc = (1.0*NB_INSTANCES)/dcalc/1E6;
     Vtot  = (1.0*NB_INSTANCES)/dtot/1E6;
     BWcomm = (sizeof(T_real)*NB_DIMS*NB_INSTANCES + 
               sizeof(int)*NB_INSTANCES + 
               sizeof(T_real)*NB_CLUSTERS*NB_DIMS)/dcomm/1E9;
     
   } else {
     tcpu1 = omp_get_wtime();
     cpu_Kmeans();                  // Run k-means on CPU
     tcpu2 = omp_get_wtime();
     dtot += tcpu2 - tcpu1;         // CPU total execution time
     
     // Speed computations
     Vtot  = (1.0*NB_INSTANCES)/dtot/1E6;
   }

   OutputResult();

   // Performance printing
   fprintf(stdout,"\nPerformances:\n");
   if (OnGPUFlag) {
     fprintf(stdout,"  Complete k-means:\n");
     fprintf(stdout,"   - Elapsed time = %.3e (s)\n", (float) dtot);
     fprintf(stdout,"   - Mptps        = %.3f \n", (float) Vtot);
     fprintf(stdout,"  Kernel computation:\n");
     fprintf(stdout,"   - Elapsed time = %.3e (s)\n", (float) dcalc);
     fprintf(stdout,"   - Mptps        = %.3f \n", (float) Vcalc);
     fprintf(stdout,"  Data transfers:\n");
     fprintf(stdout,"   - Elapsed time = %.3e (s)\n", (float) dcomm);
     fprintf(stdout,"   - BW           = %.3f (GB/s)\n", (float) BWcomm);
   } else {
     fprintf(stdout,"  Complete k-means:\n");
     fprintf(stdout,"   - Elapsed time = %.3e (s)\n", (float) dtot);
     fprintf(stdout,"   - Mptps = %.3f \n", (float) Vtot);
   }

   return(EXIT_SUCCESS);
}
