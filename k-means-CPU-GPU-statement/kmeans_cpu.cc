////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <float.h>

#include "main.h"
#include "init.h"
#include "kmeans_cpu.h"


void cpu_Kmeans()
{
  // End criteria variables
  double tolerance;        // tolerance = nb_changes / NB_INSTANCES
  int nb_iter_kmeans;      // iteration counter

  // Get k initial centroids uniformly at random
  unsigned int seed = 0;  // Get k initial centroids uniformly at random
  for (int k = 0; k < NB_CLUSTERS; k++) {
    int idx = rand_r(&seed)/((T_real)RAND_MAX) * NB_INSTANCES;
    for (int j = 0; j < NB_DIMS; j++) {
      centroid[k*NB_DIMS+j] = instance[idx*NB_DIMS + j];
    }
  }

  // Reset counters for end criteria computation
  nb_changes = 0;
  nb_iter_kmeans = 0;

  omp_set_num_threads(NbThreads); 
  #pragma omp parallel 
  {
  do {
    // Compute distances & Assign points to clusters
    #pragma omp for reduction(+: nb_changes)
    for (int i = 0; i < NB_INSTANCES; i++) {
      // - Compute distance of point i to each centroide
      int min = 0;
      T_real dist_sq;
      T_real minDist_sq = FLT_MAX;
      for (int k = 0; k < NB_CLUSTERS; k++) {
        dist_sq = 0.0;
        for (int j = 0; j < NB_DIMS; j ++) {
          dist_sq += (instance[i*NB_DIMS + j] - centroid[k*NB_DIMS + j]) *
                     (instance[i*NB_DIMS + j] - centroid[k*NB_DIMS + j]);
        }
        bool a = (dist_sq < minDist_sq);
        min = (a ? k : min);
        minDist_sq = (a ? dist_sq : minDist_sq);
      }
      // - Assign point i to its closest centroide
      if (label[i] != min) {
        nb_changes++;
        label[i] = min;
      }
    }

    // Update centroids
    // - reset counters and barycentres
    #pragma omp for
    for (int k = 0; k < NB_CLUSTERS; k++) {
      count[k] = 0;
      for (int j = 0; j < NB_DIMS; j++) {
        centroid[k*NB_DIMS + j] = 0.0;
      }
    }

    // - compute new barycenters
    #pragma omp single
    {
     for (int i = 0; i < NB_INSTANCES; i++) {
       int k = label[i];
       count[k]++;
       for (int j = 0; j < NB_DIMS; j++) {
         centroid[k*NB_DIMS + j] += instance[i*NB_DIMS + j];
       }
     }
    }
 
    #pragma omp for 
    for (int k = 0; k < NB_CLUSTERS; k++) {
      if (count[k] != 0)
        for (int j = 0; j < NB_DIMS; j++) {
          centroid[k*NB_DIMS + j] /= count[k];
        }
    }

    // Update end criteria
    #pragma omp single
    {
     tolerance = ((double) nb_changes) / NB_INSTANCES;
     //printf("Track = %llu  Tolerance = %lf\n", nb_changes, tolerance); 
     nb_changes = 0; 
     nb_iter_kmeans++;
    }
     
  } while (tolerance > TOL_KMEANS && nb_iter_kmeans < MAX_ITER_KMEANS);
  }
}
