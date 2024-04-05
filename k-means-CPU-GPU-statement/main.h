////////////////////////////////////////////////////////////////////////////////
// k-means on CPU & GPU
// S. Vialle March 2022 (with the help of G. He)
////////////////////////////////////////////////////////////////////////////////

#define DATA_FILE_NAME  "DATA_S1.txt"
#define NB_INSTANCES    1000
#define NB_DIMS         2
#define NB_CLUSTERS     4

//#define DATA_FILE_NAME  "DATA_S2.txt"
//#define NB_INSTANCES    5000
//#define NB_DIMS         2
//#define NB_CLUSTERS     15


#define OUTPUT_LABELS          "Labels_"
#define OUTPUT_CENTROIDS_FINAL "FinalCentroids_"

#define TOL_KMEANS      0.0f         // End criteria
#define MAX_ITER_KMEANS 200

#define BLOCK_SIZE_X_N  256          // BLOCK_SIZE_X related to NB_INSTANCES
#define BLOCK_SIZE_X_C  32           // BLOCK_SIZE_X related to NB_CLUSTERS

#define DEFAULT_NB_THREADS  1        // Constant for OpenMP config
#define DEFAULT_ONGPUFLAG   1
#define DEFAULT_FILE_DIRECTORY "./"

#define LABELFILEROOT       "Labels_"
#define CENTROIDFILEROOT    "Centroids_"


/*-------------------------------------------------------------------------------*/
/* Floating point datatype and op                                                */
/*-------------------------------------------------------------------------------*/
#ifdef DP
typedef double T_real;
#define T_REAL_PRINT    "%lf"
#define CUBLAS_GEAM     cublasDgeam
#define CURAND_UNIFORM  curand_uniform_double
#define REAL_MAX        DBL_MAX
#else
typedef float T_real;
#define T_REAL_PRINT    "%f"
#define CUBLAS_GEAM     cublasSgeam
#define CURAND_UNIFORM  curand_uniform
#define REAL_MAX        FLT_MAX
#endif


/*-------------------------------------------------------------------------------*/
/* Global variables                                                              */
/*-------------------------------------------------------------------------------*/
extern int NbThreads;
extern int OnGPUFlag;
extern const char *FileDirectory;
extern T_real *instance;
extern T_real *instance_T;

extern int    *label;

extern T_real *centroid;
extern unsigned long long int *count;

extern unsigned long long int nb_changes;        // Nb of pts changing of label


int main(int argc, char *argv[]);
